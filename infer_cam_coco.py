
import numpy as np
import torch
import cv2
import os
import coco.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils, visualization
import argparse
from PIL import Image
import torch.nn.functional as F
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.resnet38_SEAM_coco", type=str)
    parser.add_argument("--infer_list", default="coco/train2014.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--coco_root", default='data/coco2014', type=str)
    parser.add_argument("--out_cam", default=None, type=str)
    parser.add_argument("--out_crf", default=None, type=str) 
    parser.add_argument("--out_cam_pred", default=None, type=str)
    parser.add_argument("--out_cam_pred_alpha", default=0.26, type=float)

    args = parser.parse_args()
    crf_alpha = [1,24]
    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()
        
    infer_dataset = coco.data.COCOClsDatasetMSF(args.infer_list, coco_root=args.coco_root,
                                                  scales=[0.5, 1.0, 1.5, 2.0],
                                                  inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        model.normalize,
                                                        imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))

    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        if iter % 50 == 0:
            print('Iter: ' + str(iter) + '/' + str(len(infer_data_loader)), flush=True)
        img_name = img_name[0]; label = label[0]

        img_path = coco.data.get_img_path(img_name, args.coco_root)
        orig_img = np.asarray(Image.open(img_path).convert("RGB"))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    _, cam, _ = model_replicas[i%n_gpus](img.cuda())
                    cam = F.upsample(cam[:,1:,:,:], orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(80, 1, 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                            batch_size=12, prefetch_size=0, processes=args.num_workers)

        cam_list = thread_pool.pop_results()

        sum_cam = np.sum(cam_list, axis=0)
        sum_cam[sum_cam < 0] = 0
        cam_max = np.max(sum_cam, (1,2), keepdims=True)
        cam_min = np.min(sum_cam, (1,2), keepdims=True)
        sum_cam[sum_cam < cam_min+1e-5] = 0
        norm_cam = (sum_cam-cam_min-1e-5) / (cam_max - cam_min + 1e-5)

        cam_dict = {}
        for i in range(80):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if args.out_cam is not None:
            os.makedirs(args.out_cam, exist_ok=True)
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        if args.out_cam_pred is not None:
            bg_score = [np.ones_like(norm_cam[0])*args.out_cam_pred_alpha]
            pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
            scipy.misc.imsave(os.path.join(args.out_cam_pred, img_name + '.png'), pred.astype(np.uint8))

        def _crf_with_alpha(cam_dict, alpha):
            v = np.array(list(cam_dict.values()))
            if len(v.shape) != 3:
                return {0: np.ones((orig_img.shape[0], orig_img.shape[1]), dtype=np.float32)}
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

            n_crf_al = dict()

            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key+1] = crf_score[i+1]

            return n_crf_al

        if args.out_crf is not None:
            for t in crf_alpha:
                crf = _crf_with_alpha(cam_dict, t)
                folder = args.out_crf + ('_%.1f'%t)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                np.save(os.path.join(folder, img_name + '.npy'), crf)
