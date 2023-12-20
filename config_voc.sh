#!/usr/bin/env bash

# Download data
wget -nc http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -P ./data/
#wget -nc http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar -P ./data/
wget -nc http://pjreddie.com/media/files/VOC2012test.tar -P ./data/
wget -nc https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip -P ./data/
tar -xf ./data/VOCtrainval_11-May-2012.tar -C ./data
tar -xf ./data/VOC2012test.tar -C ./data
unzip -q ./data/SegmentationClassAug.zip -d ./data
rm -r ./data/__MACOSX
mv ./data/SegmentationClassAug ./data/VOCdevkit/VOC2012/
cp -a ./voc12/imagesets/trainaug.txt ./data/VOCdevkit/VOC2012/ImageSets/Segmentation/

# Create classification labels
python3 make_cls_labels.py
