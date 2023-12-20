import numpy as np
import torch
import random
import cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import coco.data
from tool import pyutils, imutils, torchutils, visualization, probutils
import argparse
import importlib
from tensorboardX import SummaryWriter
import torch.nn.functional as F

def setup_seed(seed):
    print("Random seed set to", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss

def max_onehot(x):
    n,c,h,w = x.size()
    x_max = torch.max(x[:,1:,:,:], dim=1, keepdim=True)[0]
    x[:,1:,:,:][x[:,1:,:,:] != x_max] = 0
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--network", default="network.resnet38_SEAM_coco", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="coco/train2014.txt", type=str)
    parser.add_argument("--val_list", default="coco/val2014.txt", type=str)
    parser.add_argument("--session_name", default="resnet38_SEAM", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--coco_root", default='data/coco2014', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    parser.add_argument("--loss_weight", default=0.2, type=float)
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--fsl_mu", default=2.5, type=float)
    parser.add_argument("--fsl_sigma", default=5.0, type=float)
    parser.add_argument('--fsl', dest='fsl', action='store_true')
    parser.add_argument('--no-fsl', dest='fsl', action='store_false')
    parser.set_defaults(fsl=True)
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    setup_seed(args.seed)
    print(vars(args))
    
    model = getattr(importlib.import_module(args.network), 'Net')()
    #tblogger = SummaryWriter(args.tblog_dir)	

    train_dataset = coco.data.COCOClsDataset(args.train_list, coco_root=args.coco_root,
                                               transform=transforms.Compose([
                        imutils.RandomResizeLong(448, 768),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray,
                        model.normalize,
                        imutils.RandomCrop(args.crop_size),
                        imutils.HWC_to_CHW,
                        torch.from_numpy
                    ]))
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn, generator=g)
    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        import network.resnet38d
        assert 'resnet38' in args.network
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_er', 'loss_ecr')

    timer = pyutils.Timer("Session started: ")
    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):

            scale_factor = 0.3
            img1 = pack[1]
            img2 = F.interpolate(img1,scale_factor=scale_factor,mode='bilinear',align_corners=True) 
            N,C,H,W = img1.size()
            label = pack[2]
            bg_score = torch.ones((N,1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)

            cam1, cam_rv1, logits1 = model(img1)
            label1 = F.adaptive_avg_pool2d(cam1, (1,1))
            loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1*label)[:,1:,:,:])
            cam1 = F.interpolate(visualization.max_norm(cam1),scale_factor=scale_factor,mode='bilinear',align_corners=True)*label
            cam_rv1 = F.interpolate(visualization.max_norm(cam_rv1),scale_factor=scale_factor,mode='bilinear',align_corners=True)*label

            cam2, cam_rv2, _ = model(img2)
            label2 = F.adaptive_avg_pool2d(cam2, (1,1))
            loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2*label)[:,1:,:,:])
            cam2 = visualization.max_norm(cam2)*label
            cam_rv2 = visualization.max_norm(cam_rv2)*label

            loss_cls1 = F.multilabel_soft_margin_loss(label1[:,1:,:,:], label[:,1:,:,:])
            loss_cls2 = F.multilabel_soft_margin_loss(label2[:,1:,:,:], label[:,1:,:,:])

            # Compute classification loss based on importance sampling
            if args.loss_weight != 0:
                probs = F.sigmoid(logits1)
                pred_is1 = probutils.sample_probs(
                    probs = probs[:, 1:].contiguous() + 1e-5,
                    num_samples = args.num_samples,
                    logits = logits1[:, 1:].contiguous())
                loss_cls_is1 = F.multilabel_soft_margin_loss(pred_is1, label[:, 1:].squeeze(-1))

            if args.loss_weight == 0:
                loss_cls1 = loss_cls1
            elif args.loss_weight == 1:
                loss_cls1 = loss_cls_is1
            else:
                loss_cls1 = (1 - args.loss_weight) * loss_cls1 + \
                             args.loss_weight * loss_cls_is1
            loss_cls = (loss_cls1 + loss_cls2)/2 + (loss_rvmin1 + loss_rvmin2)/2
            
            # Compute feature similarity loss
            if args.fsl:
                n, c, h, w = logits1.size()
                norm_cam1 = visualization.max_norm(logits1)*label
                norm_cam1[:,0,:,:] = 1-torch.max(norm_cam1[:,1:,:,:],dim=1)[0]
                img1_s = F.interpolate(img1, (h, w), mode='bilinear', align_corners=True).cuda(non_blocking=True)
                mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1_s.device)
                std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1_s.device)
                img1_s = img1_s*std_t + mean_t
                loss_fs1 = probutils.feature_similarity_loss(img1_s, norm_cam1, args.fsl_mu, args.fsl_sigma)
            else:
                loss_fs1 = torch.tensor(0, dtype=torch.float32)

            ns,cs,hs,ws = cam2.size()
            loss_er = torch.mean(torch.abs(cam1[:,1:,:,:]-cam2[:,1:,:,:]))
            #loss_er = torch.mean(torch.pow(cam1[:,1:,:,:]-cam2[:,1:,:,:], 2))
            cam1[:,0,:,:] = 1-torch.max(cam1[:,1:,:,:],dim=1)[0]
            cam2[:,0,:,:] = 1-torch.max(cam2[:,1:,:,:],dim=1)[0]
#            with torch.no_grad():
#                eq_mask = (torch.max(torch.abs(cam1-cam2),dim=1,keepdim=True)[0]<0.7).float()
            tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)#*eq_mask
            tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)#*eq_mask
            loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns,-1), k=(int)(81*hs*ws*0.2), dim=-1)[0])
            loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns,-1), k=(int)(81*hs*ws*0.2), dim=-1)[0])
            loss_ecr = loss_ecr1 + loss_ecr2

            loss = loss_cls + loss_er + loss_ecr + 0.2*loss_fs1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({'loss': loss.item(), 'loss_cls': loss_cls.item(), 'loss_er': loss_er.item(), 'loss_ecr': loss_ecr.item()})

            if (optimizer.global_step - 1) % 50 == 0:

                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step-1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'loss_cls', 'loss_er', 'loss_ecr'),
                      'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()

                # Visualization for training process
                #img_8 = img1[0].numpy().transpose((1,2,0))
                #img_8 = np.ascontiguousarray(img_8)
                #mean = (0.485, 0.456, 0.406)
                #std = (0.229, 0.224, 0.225)
                #img_8[:,:,0] = (img_8[:,:,0]*std[0] + mean[0])*255
                #img_8[:,:,1] = (img_8[:,:,1]*std[1] + mean[1])*255
                #img_8[:,:,2] = (img_8[:,:,2]*std[2] + mean[2])*255
                #img_8[img_8 > 255] = 255
                #img_8[img_8 < 0] = 0
                #img_8 = img_8.astype(np.uint8)

                #input_img = img_8.transpose((2,0,1))
                #h = H//4; w = W//4
                #p1 = F.interpolate(cam1,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                #p2 = F.interpolate(cam2,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                #p_rv1 = F.interpolate(cam_rv1,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                #p_rv2 = F.interpolate(cam_rv2,(h,w),mode='bilinear')[0].detach().cpu().numpy()

                #image = cv2.resize(img_8, (w,h), interpolation=cv2.INTER_CUBIC).transpose((2,0,1))
                #CLS1, CAM1, _, _ = visualization.generate_vis(p1, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                #CLS2, CAM2, _, _ = visualization.generate_vis(p2, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                #CLS_RV1, CAM_RV1, _, _ = visualization.generate_vis(p_rv1, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                #CLS_RV2, CAM_RV2, _, _ = visualization.generate_vis(p_rv2, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                ##MASK = eq_mask[0].detach().cpu().numpy().astype(np.uint8)*255
                #loss_dict = {'loss':loss.item(), 
                #             'loss_cls':loss_cls.item(),
                #             'loss_er':loss_er.item(),
                #             'loss_ecr':loss_ecr.item()}
                #itr = optimizer.global_step - 1
                #tblogger.add_scalars('loss', loss_dict, itr)
                #tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], itr)
                #tblogger.add_image('Image', input_img, itr)
                ##tblogger.add_image('Mask', MASK, itr)
                #tblogger.add_image('CLS1', CLS1, itr)
                #tblogger.add_image('CLS2', CLS2, itr)
                #tblogger.add_image('CLS_RV1', CLS_RV1, itr)
                #tblogger.add_image('CLS_RV2', CLS_RV2, itr)
                #tblogger.add_images('CAM1', CAM1, itr)
                #tblogger.add_images('CAM2', CAM2, itr)
                #tblogger.add_images('CAM_RV1', CAM_RV1, itr)
                #tblogger.add_images('CAM_RV2', CAM_RV2, itr)

        else:
            print('')
            timer.reset_stage()

    ckpt_name = args.session_name + '.pth'
    ckpt_dir = os.path.dirname(ckpt_name)
    if ckpt_dir != '':
        os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.module.state_dict(), ckpt_name)
