import json
import os
from tqdm import tqdm  # 显示进度
from xmltodict import unparse


# BBOX_OFFSET: Switch between 0-based and 1-based bbox.
# The COCO dataset is in 0-based format, while the VOC dataset is 1-based.
# To keep 0-based, set it to 0. To convert to 1-based, set it to 1.

def base_dict(filename, width, height, depth=3):
    return {
        "annotation": {
            "filename": os.path.split(filename)[-1],
            "folder": "VOC2COCO", "segmented": "0", "owner": {"name": "unknown"},
            "source": {'database': "The COCO 2017 database", 'annotation': "COCO 2017", "image": "unknown"},
            "size": {'width': width, 'height': height, "depth": depth},
            "object": []
        }
    }


def base_object(size_info, name, bbox, BBOX_OFFSET, img_out_size=(0, 0)):
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h

    width = size_info['width']
    height = size_info['height']

    x1 = max(x1, 0) + BBOX_OFFSET
    y1 = max(y1, 0) + BBOX_OFFSET
    x2 = min(x2, width - 1) + BBOX_OFFSET
    y2 = min(y2, height - 1) + BBOX_OFFSET
    if (img_out_size != (0, 0)):
        expand_rate_width = img_out_size[0] / width
        expand_rate_height = img_out_size[1] / height
        x1 = expand_rate_width * x1
        x2 = expand_rate_width * x2
        y1 = expand_rate_height * y1
        y2 = expand_rate_height * y2

    return {
        'name': name, 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0',
        'bndbox': {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
    }


def transform(sets=None, img_out_size=(0, 0), BBOX_OFFSET=0, parent_path="data"):
    dst_base = os.path.join(parent_path, "VOC2COCO")
    dst_dirs = {x: os.path.join(dst_base, x) for x in ["Annotations", "ImageSets", "JPEGImages"]}
    for k, d in dst_dirs.items():
        os.makedirs(d, exist_ok=True)
    labels_map = {x['id']: x['name'] for x in json.load(open(sets["test"]))['categories']}
    with open(os.path.join(dst_dirs["ImageSets"], "labels.txt"), "w") as f:
        f.writelines(list(map(lambda x: str(x) + "\n", labels_map.values())))
    for stage, filename in sets.items():
        print('Parse', stage, filename)
        data = json.load(open(filename))

        images = {}
        for src in (tqdm(data['images'], 'Start Parse Image')):
            images[src['id']] = base_dict(src['coco_url'], src['width'], src['height'], 3)

        for an in tqdm(data['annotations'], 'Start Parse Annotations'):
            ann = base_object(images[an['image_id']]['annotation']["size"], labels_map[an['category_id']], an['bbox'],
                              BBOX_OFFSET, img_out_size)
            images[an['image_id']]['annotation']['object'].append(ann)

        for key, im in tqdm(images.items(), "Write Annotations"):
            im['annotation']['object'] = im['annotation']['object'] or [None]
            # 000000041488
            # print(im)
            # if(im['annotation']['filename'] == '000000041488.jpg'):
            #     print(im['annotation']['object'])
            #     print(len(im['annotation']['object']))
            #     print(im['annotation']['object'] == [None])


            if (im['annotation']['object'] == [None]):
                continue
            os.makedirs(os.path.join(dst_dirs["Annotations"], stage), exist_ok=True)
            unparse(im, open(os.path.join(dst_dirs["Annotations"], stage, "{}.xml".format(str(key).zfill(12))), "w"),
                    full_document=False, pretty=True)
        print("Write image sets")
        with open(os.path.join(dst_dirs["ImageSets"], "{}.txt".format(stage)), "w") as f:
            f.writelines(list(map(lambda x: str(x).zfill(12) + "\n", images.keys())))




if __name__ == '__main__':
    datasets = {
        "train": "/media/heolis/967EC257F5104FE6/cocod/annotations_trainval2017/annotations/instances_train2017.json",
        "test": "/media/heolis/967EC257F5104FE6/cocod/annotations_trainval2017/annotations/instances_val2017.json"
    }
    transform(sets=datasets, img_out_size=(416, 416), parent_path="../data")
