#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid
import shutil

import numpy as np
from PIL import Image

import labelme

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "Visualization"))
    print("Creating dataset:", args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[
            dict(
                url=None,
                id=0,
                name=None,
            )
        ],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(
                supercategory=None,
                id=class_id,
                name=class_name,
            )
        )

    out_ann_file = osp.join(args.output_dir, "annotations.json")

    # 모든 이미지 파일 찾기 (jpg, jpeg, png, bmp)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(osp.join(args.input_dir, ext)))
        image_files.extend(glob.glob(osp.join(args.input_dir, "*", ext)))

    print(f"Found {len(image_files)} image files")

    annotation_id = 0

    for image_id, image_file in enumerate(image_files):
        print(f"Processing image {image_id + 1}/{len(image_files)}: {osp.basename(image_file)}")

        base = osp.splitext(osp.basename(image_file))[0]
        json_file = osp.join(osp.dirname(image_file), base + ".json")

        # 이미지 파일 복사
        if image_file.lower().endswith('.bmp'):
            out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".bmp")
        else:
            out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")

        # 이미지 정보 가져오기
        if osp.exists(json_file):
            # JSON 파일이 있는 경우 - labelme 방식으로 처리
            try:
                label_file = labelme.LabelFile(filename=json_file)
                img = labelme.utils.img_data_to_arr(label_file.imageData)
                Image.fromarray(img).save(out_img_file)
                has_annotations = True
            except Exception as e:
                print(f"Error loading labelme file {json_file}: {e}")
                # labelme 파일 로딩 실패시 원본 이미지 사용
                shutil.copy2(image_file, out_img_file)
                img = np.array(Image.open(image_file))
                has_annotations = False
                label_file = None
        else:
            # JSON 파일이 없는 경우 - 원본 이미지만 복사
            shutil.copy2(image_file, out_img_file)
            img = np.array(Image.open(image_file))
            has_annotations = False
            label_file = None

        # 이미지 정보를 COCO 데이터에 추가
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        # annotation 처리
        if has_annotations and label_file and hasattr(label_file, 'shapes'):
            masks = {}  # for area
            segmentations = collections.defaultdict(list)  # for segmentation

            for shape in label_file.shapes:
                points = shape["points"]
                label = shape["label"]
                group_id = shape.get("group_id")
                shape_type = shape.get("shape_type", "polygon")

                # 사각형인 경우 좌표 정렬 (shape_to_mask 호출 전에)
                if shape_type == "rectangle":
                    (x1, y1), (x2, y2) = points
                    # 좌표 정렬: x1,y1이 좌상단, x2,y2가 우하단이 되도록
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    # 정렬된 좌표로 points 업데이트
                    points = [(x1, y1), (x2, y2)]

                # 이제 정렬된 points로 mask 생성
                mask = labelme.utils.shape_to_mask(img.shape[:2], points, shape_type)

                if group_id is None:
                    group_id = uuid.uuid1()

                instance = (label, group_id)

                if instance in masks:
                    masks[instance] = masks[instance] | mask
                else:
                    masks[instance] = mask

                if shape_type == "rectangle":
                    (x1, y1), (x2, y2) = points
                    # 이미 정렬되어 있으므로 다시 정렬할 필요 없음
                    points = [x1, y1, x2, y1, x2, y2, x1, y2]
                elif shape_type == "circle":
                    (x1, y1), (x2, y2) = points
                    r = np.linalg.norm([x2 - x1, y2 - y1])
                    # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
                    # x: tolerance of the gap between the arc and the line segment
                    n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                    i = np.arange(n_points_circle)
                    x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                    y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                    points = np.stack((x, y), axis=1).flatten().tolist()
                else:
                    points = np.asarray(points).flatten().tolist()

                segmentations[instance].append(points)

            segmentations = dict(segmentations)

            for instance, mask in masks.items():
                cls_name, group_id = instance
                if cls_name not in class_name_to_id:
                    continue
                cls_id = class_name_to_id[cls_name]

                mask = np.asfortranarray(mask.astype(np.uint8))
                mask = pycocotools.mask.encode(mask)
                area = float(pycocotools.mask.area(mask))
                bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                data["annotations"].append(
                    dict(
                        id=annotation_id,
                        image_id=image_id,
                        category_id=cls_id,
                        segmentation=segmentations[instance],
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    )
                )
                annotation_id += 1

            # Visualization for images with annotations (simplified)
            if not args.noviz:
                out_viz_file = osp.join(args.output_dir, "Visualization", base + ".jpg")
                Image.fromarray(img).save(out_viz_file)
        else:
            # annotation이 없는 이미지의 경우 - 빈 annotation으로 처리
            print(f"  No annotations found for {osp.basename(image_file)}")

            # Visualization for images without annotations
            if not args.noviz:
                out_viz_file = osp.join(args.output_dir, "Visualization", base + ".jpg")
                Image.fromarray(img).save(out_viz_file)

    print(f"Total images processed: {len(image_files)}")
    print(f"Total annotations created: {annotation_id}")

    with open(out_ann_file, "w") as f:
        json.dump(data, f)

    print(f"COCO dataset created successfully at: {args.output_dir}")
    print(f"Annotations file: {out_ann_file}")


if __name__ == "__main__":
    main()