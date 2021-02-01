import os
import cv2
import torch
import torchvision
import PIL.Image as Image
import maskrcnn_benchmark.utils.zipreader as zipreader

from maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

from torch.utils.data import Dataset
from pycocotools.coco import COCO

import numpy as np
import ipdb


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


def keypoints_to_heat_map(keypoints, rois, heatmap_size):
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid

class COCODataset(torchvision.datasets.coco.CocoDetection):
# class COCODataset(Dataset):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms


    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)
        print("img = ", img)
        print("anno = ", anno)
        # ipdb.set_trace()

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        print("after filter crowd annotations, anno = ", anno)
        for k, ann in enumerate(anno):
            # print("ann keypoints = ", ann['keypoints'])
            # ipdb.set_trace()
            joints_3d = np.array(ann['keypoints']).reshape(17, 3)  ##17代表关键点个数，3代表(x, y, v)，v为{0:不存在, 1:存在但不可见, 2:存在且可见}
            x, y, w, h = list(map(int, np.array(ann['bbox'])))
        keypoints_to_heat_map(keypoints, rois, heatmap_size)
        ipdb.set_trace()

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # if anno and "segmentation" in anno[0]:
        #     masks = [obj["segmentation"] for obj in anno]
        #     masks = SegmentationMask(masks, img.size, mode='poly')
        #     target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            print("keypoints = ", keypoints)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx


    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


if __name__ == "__main__":
    img_ann = 'D:/AI/AIData/coco/annotations/person_keypoints_val2017.json'
    img_dir = 'D:/AI/AIData/coco//images/val2017'

    remove_images_without_annotations = True  ##有人的标注信息的图片有2693张
    coco = COCODataset(img_ann, img_dir, remove_images_without_annotations, transforms=None)
    print("len self.ids = ", len(coco.ids))
    ipdb.set_trace()

    img, target, idx = coco.__getitem__(1)
    print("target = ", target)
    ipdb.set_trace()

    coco = COCO(annotation_file=img_ann)
    catIds = coco.getCatIds(catNms=['person'])  ##person是第1个类别
    # print("catIds = ", catIds)
    # ipdb.set_trace()
    imgIds = coco.getImgIds(catIds=catIds)  ##根据catIds对imgIds进行过滤
    # print("imgIds = ", imgIds)
    # ipdb.set_trace()
    img = coco.loadImgs(imgIds[24])  ##图片在imgs中的信息
    # print("img = ", img)
    # ipdb.set_trace()
    img_name = img[0]['file_name']
    img_path = img_dir + img_name
    annIds = coco.getAnnIds(imgIds=img[0]['id'], catIds=catIds, iscrowd=None)
    print("annIds = ", annIds)
    # ipdb.set_trace()
    anns = coco.loadAnns(annIds)
    # print("anns = ", anns)
    # ipdb.set_trace()
    img_rgb = cv2.imread(img_path)
    # cv2.imshow(img_name, img_rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # ipdb.set_trace()
    # ann = anns[0]
    for k, ann in enumerate(anns):
        # print("ann keypoints = ", ann['keypoints'])
        # ipdb.set_trace()
        joints_3d = np.array(ann['keypoints']).reshape(17, 3)  ##17代表关键点个数，3代表(x, y, v)，v为{0:不存在, 1:存在但不可见, 2:存在且可见}
        x, y, w, h = list(map(int, np.array(ann['bbox'])))
        croped_person = img_rgb[y:y + h, x:x + w]
        croped_person = cv2.resize(croped_person, (WIDTH, HEIGHT)) / 5  ##将图片变暗，显示时突出关键点
        enlarge_ratio_h, enlarge_ratio_w = HEIGHT / h, WIDTH / w

        nonzero_x = joints_3d[..., 0].nonzero()[0]  ##array，x坐标不为0的关键点的index组成的array
        nonzero_y = joints_3d[..., 1].nonzero()[0]  ##array，y坐标不为0的关键点的index组成的array
        joints_3d[..., 0][nonzero_x] = (joints_3d[..., 0][
                                            nonzero_x] - x) * enlarge_ratio_w  ##各个关键点在大小为[WIDTH, HEIGHT]的resize图上的位置的x坐标
        joints_3d[..., 1][nonzero_y] = (joints_3d[..., 1][
                                            nonzero_x] - y) * enlarge_ratio_h  ##各个关键点在大小为[WIDTH, HEIGHT]的resize图上的位置的y坐标

        cv2.imwrite("./croped_person{}.png".format(k), croped_person)
        joints_3d_visible = np.zeros((17, 3))
        for i, kps in enumerate(joints_3d):
            v = kps[-1]
            if v > 0:
                joints_3d_visible[i] = np.array([1, 1, 1])
        bbox = ann['bbox']
        x1, y1, w, h = bbox
        cfg = {'image_size': np.array([WIDTH, HEIGHT]), 'num_joints': 17,
               'heatmap_size': np.array([heatmap_w, heatmap_h])}
        #  result = generate_guassian_heatmap(cfg, joints_3d, joints_3d_visible)
        targets, targets_visible = _generate_target(cfg, joints_3d, joints_3d_visible)  # 包含了17个heatmap
        heatmap = np.zeros((heatmap_w, heatmap_h))

        for i in range(len(targets)):
            heatmap += targets[i]
        xx = heatmap.astype(np.uint8)
        xx = xx * 255
        rgb = cv2.cvtColor(xx, cv2.COLOR_GRAY2RGB)
        # 通过高斯核生成的heatmap + 原始croped resized图片
        #  rgb_merged = rgb + croped_person
        #  cv2.imwrite("./heatmaps/heatmap{}.png".format(k), rgb_merged)

        heatmaps = targets
        # 将heatmap中最大激活点的位置反向映射到原图中
        resize_ratio = (enlarge_ratio_w, enlarge_ratio_h)
        splased_img = splash_per_heatmap(img_rgb, heatmaps, bbox,
                                         resize_ratio)  ##将gt keypoint生成的heatmap中最大激活点的位置反向映射到原图中

    cv2.imwrite("./maped_img.png", splased_img)
    img = splash_coco_kps(img_path, anns)  ##原始图片+边框+关键点
    cv2.imwrite("im.png", img)
    print(img.shape)
