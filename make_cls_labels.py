import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='voc')
    parser.add_argument('--train_list', default='voc12/train_aug.txt', type=str)
    parser.add_argument('--val_list', default='voc12/val.txt', type=str)
    parser.add_argument('--out', default='data/VOCdevkit/VOC2012/cls_labels.npy', type=str)
    parser.add_argument('--data_root', default='data/VOCdevkit/VOC2012', type=str)
    args = parser.parse_args()

    if args.dataset == 'voc':
        import voc12.data as data
    elif args.dataset == 'coco':
        import coco.data as data
    else:
        raise ValueError('Dataset "{}" not implemented'.format(args.dataset))

    img_name_list = data.load_img_name_list(args.train_list)
    img_name_list.extend(data.load_img_name_list(args.val_list))
    if args.dataset == 'voc':
        label_list = data.load_image_label_list_from_xml(img_name_list, args.data_root)
    if args.dataset == 'coco':
        label_list = data.load_image_label_list_from_anns(img_name_list, args.data_root)

    d = dict()
    for img_name, label in zip(img_name_list, label_list):
        d[img_name] = label

    np.save(args.out, d)