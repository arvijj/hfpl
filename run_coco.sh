#!/bin/bash

echo Starting CAM training
python3 train_cam_coco.py \
    --weights pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
    --session_name exp/ckpts/cam \
    --tblog_dir exp/logs/cam \
    "${@:1}"

echo Starting CAM inference on validation set
python3 infer_cam_coco.py \
    --weights exp/ckpts/cam.pth \
    --infer_list coco/val2014.txt \
    --out_cam exp/plabels/val/cam \
    --out_crf exp/plabels/val/crf/alpha

echo Starting CAM evaluation on validation set
python3 eval_cam_coco.py \
    --list data/coco2014/val2014.txt \
    --predict_dir exp/plabels/val/cam \
    --gt_dir data/coco2014/SegmentationClass \
    --logfile exp/logs/evallog_seam_cam_val.txt \
    --comment comment \
    --type npy \
    --curve True

echo Computing contour F-score for CAMs on validation set
python3 contour_fscore.py \
    --pred_path exp/plabels/val/cam \
    --img_list data/coco2014/val2014.txt \
    --gt_path data/coco2014/SegmentationClass \
    --is_cam \
    --is_seam \
    --num_classes 80 \
    --no-verbose

echo Starting CAM inference on training set
python3 infer_cam_coco.py \
    --weights exp/ckpts/cam.pth \
    --infer_list coco/train2014.txt \
    --out_cam exp/plabels/train/cam \
    --out_crf exp/plabels/train/crf/alpha

echo Starting AffinityNet training
python3 train_aff_coco.py \
    --weights pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
    --la_crf_dir exp/plabels/train/crf/alpha_1.0 \
    --ha_crf_dir exp/plabels/train/crf/alpha_24.0 \
    --session_name exp/ckpts/aff \
    --max_epoches 8

echo Starting AffinityNet inference on training set
python3 infer_aff_coco.py \
    --weights exp/ckpts/aff.pth \
    --infer_list coco/train2014.txt \
    --cam_dir exp/plabels/train/cam \
    --out_rw exp/plabels/train/rw

echo Starting AffinityNet evaluation on training set
python3 eval_cam_coco.py \
    --list data/coco2014/train2014.txt \
    --predict_dir exp/plabels/train/rw \
    --gt_dir data/coco2014/SegmentationClass \
    --logfile exp/logs/evallog_seam_aff_train.txt \
    --comment comment \
    --type png

echo Computing contour F-score for AffinityNet on training set
python3 contour_fscore.py \
    --pred_path exp/plabels/train/rw \
    --img_list data/coco2014/train2014.txt \
    --gt_path data/coco2014/SegmentationClass \
    --no-is_cam \
    --num_classes 80 \
    --no-verbose

echo Starting final training
python3 train_final_coco.py

echo Starting final evaluation on validation set
python3 eval_final_coco.py \
    --period val

echo Computing final contour F-score on validation set
python3 contour_fscore.py \
    --pred_path exp/results/val \
    --img_list data/coco2014/val2014.txt \
    --gt_path data/coco2014/SegmentationClass \
    --no-is_cam \
    --num_classes 80 \
    --no-verbose
