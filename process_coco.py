import argparse
import cv2
import numpy as np
import os

from coco.cocoapi.linux.PythonAPI.pycocotools.coco import COCO
from coco.cocoapi.linux.PythonAPI.pycocotools import mask as maskUtils

def main(args):
    
    # Initialize the COCO api for instance annotations and get all image ids
    ann_file = '{}/annotations/instances_{}.json'.format(args.data_dir, args.data_type)
    coco = COCO(ann_file)
    cats = dict()
    id_map = dict() # maps category id (1 to 90) to index (0 to 79)
    id_prev = 0
    for i, cat in enumerate(coco.loadCats(coco.getCatIds())):
        cats[cat['id']] = cat['name']
        id_map[cat['id']] = i
        assert cat['id'] > id_prev # make sure that the ids are in ascending order
        id_prev = cat['id']
    imgIds = coco.getImgIds()
    
    # Create folder for segmentation annotations
    if args.gen_seg_masks:
        ann_path = os.path.join(args.data_dir, 'SegmentationClass')
        if not os.path.exists(ann_path):
            os.makedirs(ann_path)
    
    # Loop through all images
    f = open(os.path.join(args.data_dir, args.data_type + '.txt'), 'w+')
    coco_imgs = coco.loadImgs(imgIds)
    for i, coco_img in enumerate(coco_imgs):
        ann_filename = str(coco_img['id']).zfill(12) + '.png'
        
        # Save matching image and annotation filenames to file
        #f.write('/' + args.data_type + '/' + coco_img['file_name'] + ' ' +
        #        '/' + 'SegmentationClass/' + ann_filename + '\n')

        # Save image IDs
        f.write(str(coco_img['id']).zfill(12) + '\n')
        
        # Create segmentation annotation image
        if args.gen_seg_masks:
            ann_img = np.zeros((coco_img['height'], coco_img['width']), dtype=np.uint8)
            annIds = coco.getAnnIds(imgIds=coco_img['id'])
            anns = coco.loadAnns(annIds)
            for ann in anns:
                ann_img = overlay_poly(ann_img, ann, id_map[ann['category_id']] + 1)
            cv2.imwrite(os.path.join(ann_path, ann_filename), ann_img)
        
        print(i+1, '/', len(coco_imgs), end='\r')
    print(len(coco_imgs), '/', len(coco_imgs))
    f.close()


# Overlay polygon
def overlay_poly(img, ann, color):
    poly = ann['segmentation']
    if isinstance(poly, dict):
        img = overlay_poly_rle(img, ann, color)
        return img
    for pol in poly:
        p = np.array(pol, dtype=np.int32)
        p = np.reshape(p, (1, -1, 2))
        cv2.fillPoly(img, p, color)
    return img


def overlay_poly_rle(img, ann, color):
    if type(ann['segmentation']['counts']) == list:
        rle = maskUtils.frPyObjects(
            [ann['segmentation']],
            ann['segmentation']['counts'][0],
            ann['segmentation']['counts'][1])
    else:
        rle = [ann['segmentation']]
    m = maskUtils.decode(rle)[:, :, 0]
    img = m * color + (1 - m) * img
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data/coco2014/', type=str)
    parser.add_argument("--data_type", default='val2017', type=str)
    parser.add_argument("--gen_seg_masks", dest='gen_seg_masks', action='store_true')
    parser.add_argument("--no-gen_seg_masks", dest='gen_seg_masks', action='store_false')
    parser.set_defaults(gen_seg_masks=True)
    args = parser.parse_args()
    main(args)
