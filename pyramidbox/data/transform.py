from __future__ import division
from __future__ import absolute_import

import random
import numpy as np
import mxnet as mx
from gluoncv.data.transforms import bbox as gbbox
from gluoncv.data.transforms import image as gimage
from gluoncv.data.transforms.experimental.image import random_color_distort
import cv2
import math
import types
from mxnet.gluon.data.vision import transforms

cv2.setNumThreads(0)

__all__ = ['PyramidBoxTrainTransform', 'PyramidBoxValTransform']


class PyramidBoxTrainTransform(object):
    """PyramidBox train transform which includes tons of image augmentations
    
    Parameters
    ----------
    width : int
        Image width
    height : int
        Image height
    anchors :list of mxnet.nd.NDAraay,optional
        Anchors generated from Pyramidbox networks,the shape must be ``(1,N,4)``
        Since anchors are shared in the entire batch so it is ``1`` for the first dimension
        ``N`` is the number of anchors for each image.
        it should be a list contain 3 items, face anchors, head anchors, body anchors
        .. hint::
            
            If anchors is ``None``, the transformation will not generate training targets.
            OtherWise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.
    
    iou_thresh : float
        IOU overlap threshold for compensate matching,default is(0.35,0.1)
    
    topk : int
        Minimum number of match per label for compensate matching,default is 6
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
        
    """

    def __init__(self, width, height, anchors=None, iou_thresh=0.35, topk=6,
                 mean=(0.485, 0.458, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        self._width = width
        self._height = height
        self._anchors = anchors
        self._mean = mean
        self._std = std
        self.random_baiducrop = RandomBaiduCrop()
        # self.transformer = transforms.Compose([transforms.ToTensor(),
        #                                         transforms.Normalize(self._mean, self._std)])
        if anchors is None:
            return
        assert isinstance(anchors, list) and len(
            anchors) == 3, "Anchors should be a list contain 3 items, face anchors, head anchors, body anchors"
        from ..nn import PyramidBoxTargetGenerator
        # @TODO: implement compensate match here
        # since we do noet have predictions yet, so we ignore sampling here
        self._target_generator = PyramidBoxTargetGenerator(iou_thresh, negative_mining_ratio=-1, **kwargs)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        """color distort"""
        # img = random_color_distort(src)

        # print("previous label shape = ", label.shape)
        target = np.zeros(shape=(label.shape[0],))

        """Pyramid Anchor sampling"""
        img, boxes, label = self.random_baiducrop(src, label[:, :4], target)
        # print("label shape = ", label.shape)
        # print('boxes shape =', boxes.shape)
        bbox = boxes
        # img = mx.nd.array(img)

        """color distort"""
        img = mx.nd.array(img)
        img = random_color_distort(img)

        # """random crop, keep aspect ration=1"""
        # h, w, _ = img.shape
        # bbox, crop_size = random_crop_with_constraints(label, (w, h))
        # x_offset, y_offset, new_width, new_height = crop_size
        # img = mx.image.fixed_crop(img, x_offset, y_offset, new_width, new_height)

        """resize with random interpolation"""
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = gimage.imresize(img, self._width, self._height, interp=interp)
        bbox = gbbox.resize(bbox, (w, h), (self._width, self._height))

        """random horizontal flip"""
        h, w, _ = img.shape
        img, flips = gimage.random_flip(img, px=0.5)
        bbox = gbbox.flip(bbox, (w, h), flip_x=flips[0])

        """To Tensor & Normalization"""
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox

        # @TODO: generating training target so cpu workers can help reduce the workload on gpu
        face_anchors, head_anchors, body_anchors = self._anchors
        gt_bboxes = mx.nd.array(bbox[:, :4]).expand_dims(0)
        gt_ids = mx.nd.zeros((1, gt_bboxes.shape[1], 1), dtype=gt_bboxes.dtype)

        face_cls_targets, face_box_targets, _ = self._target_generator(
            face_anchors, None, gt_bboxes, gt_ids)

        head_cls_targets, head_box_targets, _ = self._target_generator(
            head_anchors, None, gt_bboxes, gt_ids)

        body_cls_targets, body_box_targets, _ = self._target_generator(
            body_anchors, None, gt_bboxes, gt_ids)

        return img, \
               face_cls_targets[0], head_cls_targets[0], body_cls_targets[0], \
               face_box_targets[0], head_box_targets[0], body_box_targets[0]


class PyramidBoxValTransform(object):
    """Default SFD validation transform.

    Parameters
    ----------
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std
        # self.transformer = transforms.Compose([transforms.ToTensor(),
        #                                        transforms.Normalize(self._mean, self._std)])

    def __call__(self, src, label):
        """Apply transform to validation image/label."""
        # img = mx.nd.image.to_tensor(src)
        # img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        src = mx.nd.array(src)
        img = mx.nd.image.to_tensor(src)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, mx.nd.array(label, dtype=img.dtype)


def random_crop_with_constraints(bbox, size, min_scale=0.3, max_scale=1, min_object_overlap=0.95,
                                 min_aspect_ratio=0.9, max_aspect_ratio=1.1, max_trial=50, eps=1e-5):
    """Crop an image randomly with bounding box constraints.

    This data augmentation is used in training of
    Single Shot Multibox Detector [#]_. More details can be found in
    data augmentation section of the original paper.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    size : tuple
        Tuple of length 2 of image shape as (width, height).
    min_scale : float
        The minimum ratio between a cropped region and the original image.
        The default value is :obj:`0.3`.
    max_scale : float
        The maximum ratio between a cropped region and the original image.
        The default value is :obj:`1`.
    max_aspect_ratio : float
        The maximum aspect ratio of cropped region.
        The default value is :obj:`2`.
    max_trial : int
        Maximum number of trials for each constraint before exit no matter what.

    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape :obj:`(M, 4+)` where M <= N.
    tuple
        Tuple of length 4 as (x_offset, y_offset, new_width, new_height).

    """
    candidates = []
    assert max_scale == 1, "required max_scale=1 but got {}".format(max_scale)
    mis, mas, mir, mar = min_scale, max_scale, min_aspect_ratio, max_aspect_ratio
    sample_params = [
        [1, 1, 1, 1],
        [1, 1, mir, mar],
        [mis, mas, 1, 1],
        [mis, mas, mir, mar]]
    w, h = size
    for i in range(4):
        mis, mas, mir, mar = sample_params[i]
        for _ in range(max_trial):
            scale = random.uniform(mis, mas)
            aspect_ratio = random.uniform(
                max(mir, scale ** 2),
                min(mar, 1 / (scale ** 2)))
            if w >= h * aspect_ratio:
                crop_h = h * scale
                crop_w = crop_h * aspect_ratio
            else:
                crop_w = w * scale
                crop_h = crop_w / aspect_ratio
            crop_h, crop_w = int(crop_h), int(crop_w)
            crop_t = random.randrange(h - crop_h + 1)
            crop_l = random.randrange(w - crop_w + 1)
            crop_bb = np.array((crop_l, crop_t, crop_l + crop_w, crop_t + crop_h))
            iob = bbox_iob(bbox, crop_bb[np.newaxis]).flatten()
            iob = iob[iob > 0]
            if len(iob) >= bbox.shape[0] * 0.75 and iob.min() >= min_object_overlap - eps:
                if i != 3:  # 1:1:1:6
                    candidates.append((crop_l, crop_t, crop_w, crop_h))
                else:
                    candidates.extend([(crop_l, crop_t, crop_w, crop_h)] * 6)
                break

    # random select one
    while candidates:
        crop = candidates.pop(np.random.randint(0, len(candidates)))
        new_bbox = gbbox.crop(bbox, crop, allow_outside_center=False)
        if new_bbox.size < 1:
            continue
        new_crop = (crop[0], crop[1], crop[2], crop[3])
        return new_bbox, new_crop

    min_len = int(min(h, w) * random.uniform(min_scale, max_scale))
    crop_h, crop_w = min_len, min_len
    for _ in range(max_trial):
        crop_t = random.randrange(h - crop_h + 1)
        crop_l = random.randrange(w - crop_w + 1)
        crop = (crop_l, crop_t, crop_w, crop_h)
        new_bbox = gbbox.crop(bbox, crop, allow_outside_center=False)
        if new_bbox.size >= bbox.size * 0.5:
            return new_bbox, crop

    return bbox, (0, 0, w, h)


def bbox_iob(bbox_a, bbox_b):
    """Calculate Intersection-Over-Object(IOB) of two bounding boxes.
    ! differenct between Fast R-CNN bbox_overlaps and gluon-cv bbox_iou

    Parameters
    ----------
    bbox_a : numpy.ndarray, object bbox
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray, crop bbox
        An ndarray with shape :math:`(M, 4)`.

    Returns
    -------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        bounding boxes in `bbox_a` and `bbox_b`.

    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4]) + 1

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + 1, axis=1)
    return area_i / area_a[:, None]


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def fix_abnormal(image, bbox, label, image0, bbox0, label0):
    # print(image.shape)
    # print(bbox)
    if image.shape[0] < 10 or image.shape[1] < 10:
        # print('After BaiduCrop, the image is too small!')
        # print(image.shape)
        return image0, bbox0, label0
    bbox_mask = (bbox[:, 2] - bbox[:, 0] < 1.5) + (bbox[:, 3] - bbox[:, 1] < 1.5)
    if bbox_mask.any():
        # print('After BaiduCrop, the bbox is too small!')
        # print(bbox[bbox_mask,:])
        return image0, bbox0, label0
    return image, bbox, label


class RandomBaiduCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):

        self.mean = np.array([104, 117, 123], dtype=np.float32)
        self.maxSize = 12000  # max size
        self.infDistance = 9999999

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        # boxes = labels
        random_counter = 0

        boxArea = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        # argsort = np.argsort(boxArea)
        # rand_idx = random.randint(min(len(argsort),6))
        # print('rand idx',rand_idx)
        rand_idx = np.random.randint(len(boxArea))
        # rand_Side = boxArea[rand_idx] ** 0.5
        # add by yeweicai
        rand_Side = math.sqrt(abs(boxArea[rand_idx]))

        # rand_Side = min(boxes[rand_idx,2] - boxes[rand_idx,0] + 1, boxes[rand_idx,3] - boxes[rand_idx,1] + 1)

        anchors = [16, 32, 64, 128, 256, 512]
        distance = self.infDistance
        anchor_idx = 5
        for i, anchor in enumerate(anchors):
            if abs(anchor - rand_Side) < distance:
                distance = abs(anchor - rand_Side)
                anchor_idx = i

        target_anchor = np.random.choice(anchors[0:min(anchor_idx + 1, 5) + 1])
        ratio = float(target_anchor) / rand_Side
        ratio = ratio * (2 ** np.random.uniform(-1, 1))

        if int(height * ratio * width * ratio) > self.maxSize * self.maxSize:
            ratio = (self.maxSize * self.maxSize / (height * width)) ** 0.5

        # interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_methods = [cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = random.choice(interp_methods)
        image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)

        boxes[:, 0] *= ratio
        boxes[:, 1] *= ratio
        boxes[:, 2] *= ratio
        boxes[:, 3] *= ratio

        height, width, _ = image.shape

        sample_boxes = []

        xmin = boxes[rand_idx, 0]
        ymin = boxes[rand_idx, 1]
        bw = (boxes[rand_idx, 2] - boxes[rand_idx, 0] + 1)
        bh = (boxes[rand_idx, 3] - boxes[rand_idx, 1] + 1)

        w = h = 640

        for _ in range(50):
            if w < max(height, width):
                if bw <= w:
                    w_off = random.uniform(xmin + bw - w, xmin)
                else:
                    w_off = random.uniform(xmin, xmin + bw - w)

                if bh <= h:
                    h_off = random.uniform(ymin + bh - h, ymin)
                else:
                    h_off = random.uniform(ymin, ymin + bh - h)
            else:
                w_off = random.uniform(width - w, 0)
                h_off = random.uniform(height - h, 0)

            w_off = math.floor(w_off)
            h_off = math.floor(h_off)

            # convert to integer rect x1,y1,x2,y2
            rect = np.array([int(w_off), int(h_off), int(w_off + w), int(h_off + h)])

            # keep overlap with gt box IF center in sampled patch
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            # mask in all gt boxes that above and to the left of centers
            m1 = (rect[0] <= boxes[:, 0]) * (rect[1] <= boxes[:, 1])
            # mask in all gt boxes that under and to the right of centers
            m2 = (rect[2] >= boxes[:, 2]) * (rect[3] >= boxes[:, 3])
            # mask in that both m1 and m2 are true
            mask = m1 * m2

            overlap = jaccard_numpy(boxes, rect)
            # have any valid boxes? try again if not
            if not mask.any() and not overlap.max() > 0.7:
                continue
            else:
                sample_boxes.append(rect)

        if len(sample_boxes) > 0:
            choice_idx = np.random.randint(len(sample_boxes))
            choice_box = sample_boxes[choice_idx]
            # print('crop the box :',choice_box)
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            m1 = (choice_box[0] < centers[:, 0]) * (choice_box[1] < centers[:, 1])
            m2 = (choice_box[2] > centers[:, 0]) * (choice_box[3] > centers[:, 1])
            mask = m1 * m2
            current_boxes = boxes[mask, :].copy()
            current_labels = labels[mask]
            current_boxes[:, :2] -= choice_box[:2]
            current_boxes[:, 2:] -= choice_box[:2]
            if choice_box[0] < 0 or choice_box[1] < 0:
                new_img_width = width if choice_box[0] >= 0 else width - choice_box[0]
                new_img_height = height if choice_box[1] >= 0 else height - choice_box[1]
                image_pad = np.zeros((new_img_height, new_img_width, 3), dtype=float)
                image_pad[:, :, :] = self.mean
                start_left = 0 if choice_box[0] >= 0 else -choice_box[0]
                start_top = 0 if choice_box[1] >= 0 else -choice_box[1]
                image_pad[start_top:, start_left:, :] = image

                choice_box_w = choice_box[2] - choice_box[0]
                choice_box_h = choice_box[3] - choice_box[1]

                start_left = choice_box[0] if choice_box[0] >= 0 else 0
                start_top = choice_box[1] if choice_box[1] >= 0 else 0
                end_right = start_left + choice_box_w
                end_bottom = start_top + choice_box_h
                current_image = image_pad[start_top:end_bottom, start_left:end_right, :].copy()
                return fix_abnormal(current_image, current_boxes, current_labels, image, boxes, labels)
                # return current_image, current_boxes, current_labels

            current_image = image[choice_box[1]:choice_box[3], choice_box[0]:choice_box[2], :].copy()
            return fix_abnormal(current_image, current_boxes, current_labels, image, boxes, labels)
            # return current_image, current_boxes, current_labels
        else:
            return image, boxes, labels
