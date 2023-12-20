#!/usr/bin/env python
import argparse
import cv2
import math
import numpy as np
import os
import pandas as pd
import sys
from PIL import Image
from skimage.morphology import disk
from time import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--img_list', type=str, required=True)
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--is_cam', dest='is_cam', action='store_true')
    parser.add_argument('--no-is_cam', dest='is_cam', action='store_false')
    parser.set_defaults(is_cam=False)
    parser.add_argument('--is_crf', dest='is_crf', action='store_true')
    parser.add_argument('--no-is_crf', dest='is_crf', action='store_false')
    parser.set_defaults(is_crf=False)
    parser.add_argument('--is_seam', dest='is_seam', action='store_true')
    parser.add_argument('--no-is_seam', dest='is_seam', action='store_false')
    parser.set_defaults(is_seam=False)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    f = open(args.img_list)
    imgs = f.read().splitlines()
    f.close()

    # Accumulate intersections and total area for prediction and ground-truth masks
    fg_matches = [0.]*args.num_classes
    gt_matches = [0.]*args.num_classes
    n_fg = [0.]*args.num_classes
    n_gt = [0.]*args.num_classes
    for i, img in enumerate(imgs):
        if args.verbose and i % 50 == 0:
            print(i, '/', len(imgs))
        if args.is_cam or args.is_crf:
            predict_dict = np.load(os.path.join(args.pred_path, img + '.npy'), allow_pickle=True).item()
            if len(list(predict_dict.values())) == 0:
                pred = None
            else:
                h, w = list(predict_dict.values())[0].shape
                cam = np.zeros((args.num_classes + 1, h, w),np.float32)
                if args.is_cam:
                    for key in predict_dict.keys():
                        cam[key+1] = predict_dict[key]
                elif args.is_crf:
                    for key in predict_dict.keys():
                        cam[key] = predict_dict[key]
                if args.is_seam:
                    cam[0,:,:] = 0.25
                else:
                    cam[0, :, :] = 1. - np.sum(cam, axis=0)
                pred = np.argmax(cam, axis=0).astype(np.uint8)
        else:
            pred = np.array(Image.open(os.path.join(args.pred_path, img + '.png')))
        gt = np.array(Image.open(os.path.join(args.gt_path, img + '.png')))
        if pred is None:
            pred = np.zeros(gt.shape, dtype=np.uint8)
        gt[gt == 255] = 0
        for idx in range(1, args.num_classes + 1):
            g = np.zeros_like(gt)
            p = np.zeros_like(pred)
            g[gt == idx] = idx
            p[pred == idx] = idx
            fgm, gtm, nfg, ngt = f_matches(g, p, bound_th=0.008)
            fg_matches[idx-1] += fgm / 100000 # avoid overflow
            gt_matches[idx-1] += gtm / 100000 # avoid overflow
            n_fg[idx-1] += nfg / 100000 # avoid overflow
            n_gt[idx-1] += ngt / 100000 # avoid overflow

    # Compute the mean F-score over all classes
    F = 0.
    for idx in range(0, args.num_classes):

        # Compute precision and recall
        if n_fg[idx] == 0 and n_gt[idx] > 0:
            precision = 1
            recall = 0
        elif n_fg[idx] > 0 and n_gt[idx] == 0:
            precision = 0
            recall = 1
        elif n_fg[idx] == 0 and n_gt[idx] == 0:
            precision = 1
            recall = 1
        else:
            precision = fg_matches[idx] / float(n_fg[idx])
            recall = gt_matches[idx] / float(n_gt[idx])

        # Compute F measure
        if precision + recall == 0:
            F += 0
        else:
            F += 2 * precision * recall / (precision + recall)

    F /= args.num_classes
    print('mean F-score: %.3f%%' % (F * 100))

def f_matches(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute the intersection and total area of prediction and ground truth
    masks using morphological operators to speed it up. Used for computing
    F-score over all images in the dataset.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(np.bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(np.bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    fg_matches = np.sum(fg_boundary * gt_dil)
    gt_matches = np.sum(gt_boundary * fg_dil)

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    return fg_matches, gt_matches, n_fg, n_gt


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


if __name__ == '__main__':
    main()
