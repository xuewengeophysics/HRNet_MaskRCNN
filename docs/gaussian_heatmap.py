# -*- coding: utf-8 -*-
"""
@author: Ferry Liu
@data: 2020
"""
import cv2
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import ipdb;pdb=ipdb.set_trace
heatmap_w = 64
heatmap_h = 64
WIDTH = 224
HEIGHT = 224

cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, 17)]
COLORS = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]


def _generate_target(cfg, joints_3d, joints_3d_visible, sigma=3):
    """Generate the target heatmap.

    Args:
        cfg (dict): data config
        joints_3d: np.ndarray ([num_joints, 3])
        joints_3d_visible: np.ndarray ([num_joints, 3])

    Returns:
        tuple: A tuple containing targets.

        - target: Target heatmaps.
        - target_weight: (1: visible, 0: invisible)
    ??? image_size 在其中的作用, 表明关键点坐标的相对位置
    """
    num_joints = cfg['num_joints']
    image_size = cfg['image_size']
    heatmap_size = cfg['heatmap_size']
    target_weight = np.zeros((num_joints, 1), dtype=np.float32)
    target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                        dtype=np.float32)
    tmp_size = sigma * 3
    for joint_id in range(num_joints):
        heatmap_vis = joints_3d_visible[joint_id, 0]
        target_weight[joint_id] = heatmap_vis
        feat_stride = image_size / heatmap_size
        mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[
                0] < 0 or br[1] < 0:
            print("warn: {}".format(joint_id))
            target_weight[joint_id] = 0
        if target_weight[joint_id] > 0.5:
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]
            x0 = y0 = size // 2
            # The gaussian is not normalized,
            # we want the center value to equal 1
            g = np.exp(-((x - x0)**2 + (y - y0)**2) /
                        (2 * sigma**2))
            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return target, target_weight


def generate_guassian_heatmap(cfg, joints_3d, joints_3d_visible, sigma=2):
    """
       Args:
           joints_3d: np.ndarray ([num_joints, 3])
           joints_3d_visiable ([num_joints, 3])
       Return:
           tuple: A tuple containing targets.
           - target: Target heatmap.
           - target_weight: (1: visiable, 0: invisiable)
    """
    num_joints = cfg['num_joints']
    image_size = cfg['image_size']
    heatmap_size = cfg['heatmap_size']
    target_weight = np.zeros((num_joints, 1), dtype=np.float32)
    target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]))  # 注意顺序y, x
    tmp_size = sigma * 3
    for joint_id in range(num_joints):
        heatmap_vis = joints_3d_visible[joint_id, 0]  # 只有0这个位置有用
        target_weight[joint_id] = heatmap_vis
        feat_stride = image_size / heatmap_size
        print("feat_stride: ", feat_stride)
        mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
        # check that any part of the guassian is in-bounds 【原因？】
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[0] < 0 or br[1] < 0:
            print("ul:{}, br:{}; warning: joints {} the guassian is in-bounds".format(ul, br, joint_id))
            target_weight[joint_id] = 0
        if target_weight[joint_id] > 0.5:
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]  # equal to x[:, np.newaxis]
            x0 = y0 = size // 2
            # gaussian is not normalized, we want ro center value to equal 1
            g = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return target, target_weight


def splash_coco_kps(img_path, anns):
    img = cv2.imread(img_path)
    for ann in anns:
        keypoints = ann['keypoints']
        keypoints = np.array(keypoints).reshape(17, 3)
        bbox = ann['bbox']
        x, y, w, h = list(map(int, bbox))
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
        for i, kpt in enumerate(keypoints):
            if kpt[-1] == 0: continue
            x, y, _ = kpt
            p = (x, y)
            thickness = 3
            color = COLORS[i]
            img = cv2.circle(img,p,thickness,color,-1)
    return img


def splash_per_heatmap(img, heatmap, bbox, enlarge_ratio):
    x1, y1, _, _ = bbox
    for j, hm_j in enumerate(heatmap):
        hm_j = heatmap[j, :, :]
        idx = hm_j.argmax()
        y, x = np.unravel_index(idx, hm_j.shape)
        score = hm_j[y][x]  # 注意坐标顺序
        if score < 0.5: continue
        x = x/cfg['heatmap_size'][0]*cfg['image_size'][0]        
        y = y/cfg['heatmap_size'][1]*cfg['image_size'][1]
        x, y = int(x/enlarge_ratio[0]) + x1, int(y/enlarge_ratio[1]) + y1
        point = (int(x), int(y))
        #  print('point location: ', point)
        img = cv2.circle(img, point, 3, (255, 0, 0), -1)
    return img


if __name__ == "__main__":
    img_ann = 'D:/AI/AIData/coco/annotations/person_keypoints_val2017.json'
    img_dir = 'D:/AI/AIData/coco//images/val2017/'
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
        joints_3d =np.array(ann['keypoints']).reshape(17, 3)  ##17代表关键点个数，3代表(x, y, v)，v为{0:不存在, 1:存在但不可见, 2:存在且可见}
        x, y, w, h = list(map(int, np.array(ann['bbox'])))
        croped_person = img_rgb[y:y+h, x:x+w]
        croped_person = cv2.resize(croped_person, (WIDTH, HEIGHT)) / 5  ##将图片变暗，显示时突出关键点
        enlarge_ratio_h, enlarge_ratio_w = HEIGHT/h, WIDTH/w
        
        nonzero_x = joints_3d[..., 0].nonzero()[0]  ##array，x坐标不为0的关键点的index组成的array
        nonzero_y = joints_3d[..., 1].nonzero()[0]  ##array，y坐标不为0的关键点的index组成的array
        joints_3d[..., 0][nonzero_x] = (joints_3d[..., 0][nonzero_x] - x) * enlarge_ratio_w  ##各个关键点在大小为[WIDTH, HEIGHT]的resize图上的位置的x坐标
        joints_3d[..., 1][nonzero_y] = (joints_3d[..., 1][nonzero_y] - y) * enlarge_ratio_h  ##各个关键点在大小为[WIDTH, HEIGHT]的resize图上的位置的y坐标

        cv2.imwrite("./croped_person{}.png".format(k), croped_person)
        joints_3d_visible = np.zeros((17, 3))
        for i, kps in enumerate(joints_3d):
            v = kps[-1]
            if v > 0:
                joints_3d_visible[i] = np.array([1,1,1])
        bbox = ann['bbox']
        x1, y1, w, h = bbox
        cfg = {'image_size': np.array([WIDTH, HEIGHT]), 'num_joints': 17, 'heatmap_size': np.array([heatmap_w, heatmap_h])}
        #  result = generate_guassian_heatmap(cfg, joints_3d, joints_3d_visible)
        targets, targets_visible = _generate_target(cfg, joints_3d, joints_3d_visible) # 包含了17个heatmap
        ipdb.set_trace()
        heatmap = np.zeros((heatmap_w, heatmap_h))

        for i in range(len(targets)):
            heatmap += targets[i]
        xx = heatmap.astype(np.uint8)
        xx = xx * 255
        rgb = cv2.cvtColor(xx, cv2.COLOR_GRAY2RGB)
        # 通过高斯核生成的heatmap + 原始croped resized图片
        # rgb_merged = rgb + croped_person
        # cv2.imwrite("./heatmap{}.png".format(k), rgb_merged)


        heatmaps = targets
        # 将heatmap中最大激活点的位置反向映射到原图中
        resize_ratio = (enlarge_ratio_w, enlarge_ratio_h)
        splased_img = splash_per_heatmap(img_rgb, heatmaps, bbox, resize_ratio)  ##将gt keypoint生成的heatmap中最大激活点的位置反向映射到原图中

    cv2.imwrite("./maped_img.png", splased_img)
    img = splash_coco_kps(img_path, anns)  ##原始图片+边框+关键点
    cv2.imwrite("im.png", img)
    print(img.shape)
