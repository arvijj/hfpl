#!/bin/bash

echo Starting CAM training
python3 train_cam_voc.py \
    --weights pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
    --session_name exp/ckpts/cam \
    --tblog_dir exp/logs/cam \
    "${@:1}"

echo Starting CAM inference on validation set
python3 infer_cam_voc.py \
    --weights exp/ckpts/cam.pth \
    --infer_list voc12/val.txt \
    --out_cam exp/plabels/val/cam \
    --out_crf exp/plabels/val/crf/alpha

echo Starting CAM evaluation on validation set
python3 eval_cam_voc.py \
    --list data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
    --predict_dir exp/plabels/val/cam \
    --gt_dir data/VOCdevkit/VOC2012/SegmentationClass \
    --logfile exp/logs/evallog_seam_cam_val.txt \
    --comment comment \
    --type npy \
    --curve True

echo Computing contour F-score for CAMs on validation set
python3 contour_fscore.py \
    --pred_path exp/plabels/val/cam \
    --img_list data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
    --gt_path data/VOCdevkit/VOC2012/SegmentationClass \
    --is_cam \
    --is_seam \
    --num_classes 20 \
    --no-verbose

echo Starting CAM inference on training set
python3 infer_cam_voc.py \
    --weights exp/ckpts/cam.pth \
    --infer_list voc12/train_aug.txt \
    --out_cam exp/plabels/train/cam \
    --out_crf exp/plabels/train/crf/alpha

echo Starting CAM evaluation on training set
python3 eval_cam_voc.py \
    --list data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt \
    --predict_dir exp/plabels/train/cam \
    --gt_dir data/VOCdevkit/VOC2012/SegmentationClass \
    --logfile exp/logs/evallog_seam_cam_train.txt \
    --comment comment \
    --type npy \
    --curve True

echo Computing contour F-score for CAMs on training set
python3 contour_fscore.py \
    --pred_path exp/plabels/train/cam \
    --img_list data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt \
    --gt_path data/VOCdevkit/VOC2012/SegmentationClass \
    --is_cam \
    --is_seam \
    --num_classes 20 \
    --no-verbose

echo Starting AffinityNet training
python3 train_aff_voc.py \
    --weights pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
    --la_crf_dir exp/plabels/train/crf/alpha_1.0 \
    --ha_crf_dir exp/plabels/train/crf/alpha_24.0 \
    --session_name exp/ckpts/aff \
    --max_epoches 8

echo Starting AffinityNet inference on validation set
python3 infer_aff_voc.py \
    --weights exp/ckpts/aff.pth \
    --infer_list voc12/val.txt \
    --cam_dir exp/plabels/val/cam \
    --out_rw exp/plabels/val/rw

echo Starting AffinityNet evaluation on validation set
python3 eval_cam_voc.py \
    --list data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
    --predict_dir exp/plabels/val/rw \
    --gt_dir data/VOCdevkit/VOC2012/SegmentationClass \
    --logfile exp/logs/evallog_seam_aff_val.txt \
    --comment comment \
    --type png

echo Computing contour F-score for AffinityNet on validation set
python3 contour_fscore.py \
    --pred_path exp/plabels/val/rw \
    --img_list data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
    --gt_path data/VOCdevkit/VOC2012/SegmentationClass \
    --no-is_cam \
    --num_classes 20 \
    --no-verbose

echo Starting AffinityNet inference on training set
python3 infer_aff_voc.py \
    --weights exp/ckpts/aff.pth \
    --infer_list voc12/train_aug.txt \
    --cam_dir exp/plabels/train/cam \
    --out_rw exp/plabels/train/rw

echo Starting AffinityNet evaluation on training set
python3 eval_cam_voc.py \
    --list data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt \
    --predict_dir exp/plabels/train/rw \
    --gt_dir data/VOCdevkit/VOC2012/SegmentationClass \
    --logfile exp/logs/evallog_seam_aff_train.txt \
    --comment comment \
    --type png

echo Computing contour F-score for AffinityNet on training set
python3 contour_fscore.py \
    --pred_path exp/plabels/train/rw \
    --img_list data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt \
    --gt_path data/VOCdevkit/VOC2012/SegmentationClass \
    --no-is_cam \
    --num_classes 20 \
    --no-verbose

echo Starting final training
python3 train_final_voc.py

echo Starting final evaluation on validation set
python3 eval_final_voc.py \
    --period val

echo Computing final contour F-score on validation set
python3 contour_fscore.py \
    --pred_path exp/results/val \
    --img_list data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
    --gt_path data/VOCdevkit/VOC2012/SegmentationClass \
    --no-is_cam \
    --num_classes 20 \
    --no-verbose

echo Starting final evaluation on training set
python3 eval_final_voc.py \
    --period train

echo Computing final contour F-score on training set
python3 contour_fscore.py \
    --pred_path exp/results/train \
    --img_list data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt \
    --gt_path data/VOCdevkit/VOC2012/SegmentationClass \
    --no-is_cam \
    --num_classes 20 \
    --no-verbose
