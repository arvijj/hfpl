#!/usr/bin/env bash

# Download data
echo Downloading training set
wget -nc http://images.cocodataset.org/zips/train2017.zip -P ./data/
echo Downloading validation set
wget -nc http://images.cocodataset.org/zips/val2017.zip -P ./data/
echo Downloading annotations
wget -nc http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ./data/
wget -nc http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P ./data/

# Unzip data
echo Unzipping training set
unzip -n -q ./data/train2017.zip -d ./data/coco2014/
echo Unzipping validation set
unzip -n -q ./data/val2017.zip -d ./data/coco2014/
echo Unzipping annotations
unzip -n -q ./data/annotations_trainval2017.zip -d ./data/coco2014/
unzip -n -q ./data/annotations_trainval2014.zip -d ./data/coco2014/

# Create a single folder for the images
mv data/coco2014/train2017 data/coco2014/Images
mv data/coco2014/val2017/* data/coco2014/Images/
rm -r data/coco2014/val2017

# Compile COCO API
echo Compiling COCO API
cd coco/cocoapi/linux/PythonAPI/
make
cd -

# Process data
echo Processing data
python3 process_coco.py --data_type train2014 --no-gen_seg_masks
python3 process_coco.py --data_type train2017
python3 process_coco.py --data_type val2014 --no-gen_seg_masks
python3 process_coco.py --data_type val2017

# Create classification labels
echo Creating cls_labels.npy, this may take a while
python3 make_cls_labels.py \
    --dataset coco \
    --train_list coco/train2017.txt \
    --val_list coco/val2017.txt \
    --out data/coco2014/cls_labels.npy \
    --data_root data/coco2014
