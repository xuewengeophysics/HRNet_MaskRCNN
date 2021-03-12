# -*- coding: utf-8 -*-
"""
@author: Ferry Liu
@data: 2020
"""
import torch
import cv2
import numpy as np
from numpy.linalg import LinAlgError
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


def keypointrcnn_inference(pred_heamtmap, boxes):
    # type: (Tensor, List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
    kp_probs = []
    kp_scores = []

    boxes_per_image = [box.size(0) for box in boxes]
    heatmaps = pred_heamtmap.split(boxes_per_image, dim=0)

    for heatmap, bbox in zip(heatmaps, boxes):
        kp_prob, scores = heatmaps2kps(heatmap, bbox)
        kp_probs.append(kp_prob)
        kp_scores.append(scores)

    return kp_probs, kp_scores


def heatmaps2kps(heatmaps, bboxes):
    xy_preds = torch.zeros((len(heatmaps), num_joints, 3), dtype=torch.float32, device=heatmaps.device)
    end_scores = torch.zeros((len(heatmaps), num_joints), dtype=torch.float32, device=heatmaps.device)
    heatmaps = heatmaps.cpu().detach().numpy()
    bboxes = bboxes.cpu().detach().numpy()
    for k, (heatmap, bbox) in enumerate(zip(heatmaps, bboxes)):
        x1, y1, h, w = bbox
        enlarge_ratio_h, enlarge_ratio_w = heatmap_height/h, heatmap_width/w
        enlarge_ratio = [enlarge_ratio_w, enlarge_ratio_h]
        for j, hm_j in enumerate(heatmap):
            hm_j = heatmap[j, :, :]
            idx = hm_j.argmax()
            y, x = np.unravel_index(idx, hm_j.shape)
            score = hm_j[y][x]  # 注意坐标顺序
            #  if score < 0.5: continue
            x = x/cfg['heatmap_size'][0]*cfg['image_size'][0]
            y = y/cfg['heatmap_size'][1]*cfg['image_size'][1]
            x, y = x/enlarge_ratio[0] + x1, y/enlarge_ratio[1] + y1
            xy_preds[k][j][0] = x
            xy_preds[k][j][1] = y
            xy_preds[k][j][2] = float(score)
            end_scores[k][j] = float(score)
    return xy_preds, end_scores


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

def _udp_generate_target(cfg, joints_3d, joints_3d_visible):
    """Generate the target heatmap via 'UDP' approach. Paper ref: Huang et
    al. The Devil is in the Details: Delving into Unbiased Data Processing
    for Human Pose Estimation (CVPR 2020).
    Note:
        num keypoints: K
        heatmap height: H
        heatmap width: W
        num target channels: C
        C = K if target_type=='GaussianHeatMap'
        C = 3*K if target_type=='CombinedTarget'
    Args:
        cfg (dict): data config
        joints_3d (np.ndarray[K, 3]): Annotated keypoints.
        joints_3d_visible (np.ndarray[K, 3]): Visibility of keypoints.
        factor (float): kernel factor for GaussianHeatMap target or
            valid radius factor for CombinedTarget.
        target_type (str): 'GaussianHeatMap' or 'CombinedTarget'.
            GaussianHeatMap: Heatmap target with gaussian distribution.
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
    Returns:
        tuple: A tuple containing targets.
        - target (np.ndarray[C, H, W]): Target heatmaps.
        - target_weight (np.ndarray[K, 1]): (1: visible, 0: invisible)
    """
    num_joints = cfg['num_joints']
    image_size = cfg['image_size']
    heatmap_size = cfg['heatmap_size']
    # joint_weights = cfg['joint_weights']
    # use_different_joint_weights = cfg['use_different_joint_weights']

    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_3d_visible[:, 0]

    target_type = cfg['TARGET_TYPE']
    assert target_type in ['GaussianHeatMap', 'CombinedTarget']

    factor = cfg['factor']

    if target_type == 'GaussianHeatMap':
        target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = factor * 3

        # prepare for gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, None]

        for joint_id in range(num_joints):
            feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
            mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            mu_x_ac = joints_3d[joint_id][0] / feat_stride[0]
            mu_y_ac = joints_3d[joint_id][1] / feat_stride[1]
            x0 = y0 = size // 2
            x0 += mu_x_ac - mu_x
            y0 += mu_y_ac - mu_y
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * factor ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    elif target_type == 'CombinedTarget':
        target = np.zeros(
            (num_joints, 3, heatmap_size[1] * heatmap_size[0]),
            dtype=np.float32)
        feat_width = heatmap_size[0]
        feat_height = heatmap_size[1]
        feat_x_int = np.arange(0, feat_width)
        feat_y_int = np.arange(0, feat_height)
        feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
        feat_x_int = feat_x_int.flatten()
        feat_y_int = feat_y_int.flatten()
        # Calculate the radius of the positive area in classification heatmap.
        valid_radius = factor * heatmap_size[1]
        feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
        for joint_id in range(num_joints):
            mu_x = joints_3d[joint_id][0] / feat_stride[0]
            mu_y = joints_3d[joint_id][1] / feat_stride[1]
            x_offset = (mu_x - feat_x_int) / valid_radius
            y_offset = (mu_y - feat_y_int) / valid_radius
            dis = x_offset ** 2 + y_offset ** 2
            keep_pos = np.where(dis <= 1)[0]
            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id, 0, keep_pos] = 1
                target[joint_id, 1, keep_pos] = x_offset[keep_pos]
                target[joint_id, 2, keep_pos] = y_offset[keep_pos]
        target = target.reshape(num_joints * 3, heatmap_size[1], heatmap_size[0])

    # if use_different_joint_weights:
    #     target_weight = np.multiply(target_weight, joint_weights)

    return target, target_weight

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def post(coords, batch_heatmaps):
    '''
    DARK post-pocessing
    :param coords: batchsize*num_kps*2
    :param batch_heatmaps:batchsize*num_kps*high*width
    :return:
    '''

    shape_pad = list(batch_heatmaps.shape)
    shape_pad[2] = shape_pad[2] + 2
    shape_pad[3] = shape_pad[3] + 2

    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            mapij=batch_heatmaps[i,j,:,:]
            maxori = np.max(mapij)
            mapij= cv2.GaussianBlur(mapij,(7, 7), 0)
            max = np.max(mapij)
            min = np.min(mapij)
            mapij = (mapij-min)/(max-min) * maxori
            batch_heatmaps[i, j, :, :]= mapij
    batch_heatmaps = np.clip(batch_heatmaps,0.001,50)
    batch_heatmaps = np.log(batch_heatmaps)
    batch_heatmaps_pad = np.zeros(shape_pad,dtype=float)
    batch_heatmaps_pad[:, :, 1:-1,1:-1] = batch_heatmaps
    batch_heatmaps_pad[:, :, 1:-1, -1] = batch_heatmaps[:, :, :,-1]
    batch_heatmaps_pad[:, :, -1, 1:-1] = batch_heatmaps[:, :, -1, :]
    batch_heatmaps_pad[:, :, 1:-1, 0] = batch_heatmaps[:, :, :, 0]
    batch_heatmaps_pad[:, :, 0, 1:-1] = batch_heatmaps[:, :, 0, :]
    batch_heatmaps_pad[:, :, -1, -1] = batch_heatmaps[:, :, -1 , -1]
    batch_heatmaps_pad[:, :, 0, 0] = batch_heatmaps[:, :, 0, 0]
    batch_heatmaps_pad[:, :, 0, -1] = batch_heatmaps[:, :, 0, -1]
    batch_heatmaps_pad[:, :, -1, 0] = batch_heatmaps[:, :, -1, 0]
    I = np.zeros((shape_pad[0],shape_pad[1]))
    Ix1 = np.zeros((shape_pad[0], shape_pad[1]))
    Iy1 = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1y1 = np.zeros((shape_pad[0],shape_pad[1]))
    Ix1_y1_ = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1_ = np.zeros((shape_pad[0], shape_pad[1]))
    Iy1_ = np.zeros((shape_pad[0], shape_pad[1]))
    coords = coords.astype(np.int32)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            I[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1]+1, coords[i, j, 0]+1]
            Ix1[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1]+1, coords[i, j, 0] + 2]
            Ix1_[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1]+1, coords[i, j, 0] ]
            Iy1[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1] + 2, coords[i, j, 0]+1]
            Iy1_[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1] , coords[i, j, 0]+1]
            Ix1y1[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1] + 2, coords[i, j, 0] + 2]
            Ix1_y1_[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1], coords[i, j, 0]]
    dx = 0.5 * (Ix1 -  Ix1_)
    dy = 0.5 * (Iy1 - Iy1_)
    D = np.zeros((shape_pad[0],shape_pad[1],2))
    D[:,:,0]=dx
    D[:,:,1]=dy
    D.reshape((shape_pad[0],shape_pad[1],2,1))
    dxx = Ix1 - 2*I + Ix1_
    dyy = Iy1 - 2*I + Iy1_
    dxy = 0.5*(Ix1y1- Ix1 -Iy1 + I + I -Ix1_-Iy1_+Ix1_y1_)
    hessian = np.zeros((shape_pad[0],shape_pad[1],2,2))
    hessian[:, :, 0, 0] = dxx
    hessian[:, :, 1, 0] = dxy
    hessian[:, :, 0, 1] = dxy
    hessian[:, :, 1, 1] = dyy
    inv_hessian = np.zeros(hessian.shape)
    # hessian_test = np.zeros(hessian.shape)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            hessian_tmp = hessian[i,j,:,:]
            try:
                inv_hessian[i,j,:,:] = np.linalg.inv(hessian_tmp)
            except LinAlgError:
                inv_hessian[i, j, :, :] = np.zeros((2,2))
            # hessian_test[i,j,:,:] = np.matmul(hessian[i,j,:,:],inv_hessian[i,j,:,:])
            # print( hessian_test[i,j,:,:])
    res = np.zeros(coords.shape)
    coords = coords.astype(np.float)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            D_tmp = D[i,j,:]
            D_tmp = D_tmp[:,np.newaxis]
            shift = np.matmul(inv_hessian[i,j,:,:],D_tmp)
            # print(shift.shape)
            res_tmp = coords[i, j, :] -  shift.reshape((-1))
            res[i,j,:] = res_tmp
    return res


def _udp_get_final_preds(cfg, batch_heatmaps):
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    if cfg['TARGET_TYPE'] == 'GaussianHeatMap':
        coords, maxvals = get_max_preds(batch_heatmaps)
        # if config.TEST.POST_PROCESS:
        #     coords = post(coords,batch_heatmaps)
    elif cfg['TARGET_TYPE'] == 'CombinedTarget':
        net_output = batch_heatmaps.copy()
        kps_pos_distance_x = cfg['factor'] * heatmap_height
        kps_pos_distance_y = cfg['factor'] * heatmap_height
        batch_heatmaps = net_output[:,::3,:]
        offset_x = net_output[:,1::3,:] * kps_pos_distance_x
        offset_y = net_output[:,2::3,:] * kps_pos_distance_y
        for i in range(batch_heatmaps.shape[0]):
            for j in range(batch_heatmaps.shape[1]):
                batch_heatmaps[i,j,:,:] = cv2.GaussianBlur(batch_heatmaps[i,j,:,:],(15, 15), 0)
                offset_x[i,j,:,:] = cv2.GaussianBlur(offset_x[i,j,:,:],(7, 7), 0)
                offset_y[i,j,:,:] = cv2.GaussianBlur(offset_y[i,j,:,:],(7, 7), 0)
        coords, maxvals = get_max_preds(batch_heatmaps)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                px = int(coords[n][p][0])
                py = int(coords[n][p][1])
                coords[n][p][0] += offset_x[n,p,py,px]
                coords[n][p][1] += offset_y[n,p,py,px]

    preds = coords.copy()
    preds_in_input_space = preds.copy()
    preds_in_input_space[:,:, 0] = preds_in_input_space[:,:, 0] / (heatmap_width - 1.0) * (4 * heatmap_width - 1.0)
    preds_in_input_space[:,:, 1] = preds_in_input_space[:,:, 1] / (heatmap_height - 1.0) * (4 * heatmap_height - 1.0)

    return preds, maxvals, preds_in_input_space



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


def splash_keypoint(img, preds, bboxs):
    for j,batch in enumerate(preds):
        x1, y1, _, _ = bboxs[j]
        _, _, w, h = list(map(int, np.array(bboxs[j])))
        enlarge_ratio_h, enlarge_ratio_w = HEIGHT / h, WIDTH / w
        enlarge_ratio = (enlarge_ratio_w, enlarge_ratio_h)
        for k in range(batch.shape[0]):
            x = batch[k][0]
            y = batch[k][1]
            if x * y < 1.0: continue
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
    heatmaps = []
    bboxs = []
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
        cfg = {'image_size': np.array([WIDTH, HEIGHT]), 'num_joints': 17, 'heatmap_size': np.array([heatmap_w, heatmap_h]),
               'TARGET_TYPE': 'CombinedTarget', 'factor': 0.0625}
        #  result = generate_guassian_heatmap(cfg, joints_3d, joints_3d_visible)
        # targets, targets_visible = _generate_target(cfg, joints_3d, joints_3d_visible) # 包含了17个heatmap
        ##如果是'CombinedTarget'，维度为(17*3, 64, 64)；
        targets, targets_visible = _udp_generate_target(cfg, joints_3d, joints_3d_visible)  # 包含了17个heatmap
        # ipdb.set_trace()
        heatmap = np.zeros((heatmap_w, heatmap_h))

        for i in range(len(targets)):
            heatmap += targets[i]
        xx = heatmap.astype(np.uint8)
        xx = xx * 255
        rgb = cv2.cvtColor(xx, cv2.COLOR_GRAY2RGB)
        # 通过高斯核生成的heatmap + 原始croped resized图片
        # rgb_merged = rgb + croped_person
        # cv2.imwrite("./heatmap{}.png".format(k), rgb_merged)


        heatmaps.append(targets)
        bboxs.append(bbox)
        # 将heatmap中最大激活点的位置反向映射到原图中
        resize_ratio = (enlarge_ratio_w, enlarge_ratio_h)
        # splased_img = splash_per_heatmap(img_rgb, heatmaps, bbox, resize_ratio)  ##将gt keypoint生成的heatmap中最大激活点的位置反向映射到原图中

    heatmaps = np.array(heatmaps)
    preds, maxvals, preds_in_input_space = _udp_get_final_preds(cfg, heatmaps)
    # ipdb.set_trace()
    splased_img = splash_keypoint(img_rgb, preds, bboxs)
    cv2.imwrite("./udp_heatmap.png", splased_img)
    img = splash_coco_kps(img_path, anns)  ##原始图片+边框+关键点
    cv2.imwrite("im.png", img)
    print(img.shape)
