from __future__ import division
from __future__ import absolute_import

import random
import numpy as np
import mxnet as mx
from gluoncv.data.transforms import bbox as gbbox
from gluoncv.data.transforms import image as gimage
from gluoncv.data.transforms.experimental.image import random_color_distort
from mxnet.gluon.data.vision import transforms

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

    def __init__(self, width, height, anchors=None, iou_thresh=0.5, topk=6,
                 mean=(0.485, 0.458, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        self._width = width
        self._height = height
        self._anchors = anchors
        self._mean = mean
        self._std = std
        self.transformer = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(self._mean, self._std)])
        if anchors is None:
            return
        assert isinstance(anchors,list) and len(anchors)==3,"Anchors should be a list contain 3 items, face anchors, head anchors, body anchors"
        from ..nn import PyramidBoxTargetGenerator
        # @TODO: implement compensate match here
        # since we do noet have predictions yet, so we ignore sampling here
        self._target_generator = PyramidBoxTargetGenerator(iou_thresh,negative_mining_ratio=-1,**kwargs)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        """color distort"""
        img = random_color_distort(src)

        """random crop, keep aspect ration=1"""
        h, w, _ = img.shape
        bbox, crop_size = random_crop_with_constraints(label, (w, h))
        x_offset, y_offset, new_width, new_height = crop_size
        img = mx.image.fixed_crop(img, x_offset, y_offset, new_width, new_height)

        """resize with random interpolation"""
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = gimage.imresize(img, self._width, self._height, interp=interp)
        bbox = gbbox.resize(bbox, (w, h), (self._width, self._height))

        """radom horizontal flip"""
        h, w, _ = img.shape
        img, flips = gimage.random_flip(img, px=0.5)
        bbox = gbbox.flip(bbox, (w, h), flip_x=flips[0])

        """To Tensor & Normalization"""
        img = self.transformer(img)

        if self._anchors is None:
            return img, bbox

        # @TODO: generating training target so cpu workers can help reduce the workload on gpu
        face_anchors,head_anchors,body_anchors = self._anchors
        gt_bboxes = mx.nd.array(bbox[:,:4]).expand_dims(0)
        gt_ids = mx.nd.zeros((1,gt_bboxes.shape[1],1),dtype=gt_bboxes.dtype)

        face_cls_targets,face_box_targets,_ = self._target_generator(
            face_anchors,None,gt_bboxes,gt_ids)

        head_cls_targets, head_box_targets, _ = self._target_generator(
            head_anchors, None, gt_bboxes, gt_ids)

        body_cls_targets, body_box_targets, _ = self._target_generator(
            body_anchors, None, gt_bboxes, gt_ids)

        return img,\
               face_cls_targets[0],head_cls_targets[0],body_cls_targets[0],\
               face_box_targets[0],head_box_targets[0],body_box_targets[0]

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
        self.transformer = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(self._mean, self._std)])
    def __call__(self, src, label):
        """Apply transform to validation image/label."""
        # img = mx.nd.image.to_tensor(src)
        # img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        img = self.transformer(src)
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


# def random_crop_with_constraints(bbox, size, min_scale=0.3, max_scale=1, min_object_overlap=0.95,
#                                  min_aspect_ratio=0.9, max_aspect_ratio=1.1, max_trial=50, eps=1e-5):
#     """Crop an image randomly with bouding box constrain
#     Parameter
#     ----------
#     bbox : mx.ndarray.NDArray  or numpy.ndarray (N,4+)
#         Numpy.ndarray with shape (N,4+) where N is the Number of bounding boxes.
#         The second axis represents attributes of the bounding box.
#         Specifically,these are:math:`{x_{min},y_{min},x_{max},y_{max}}`
#     size : tuple
#         Tuple of length 2 of image shape as (width,height)
#     min_scale : float
#         The minimum ratio between a cropped region and the original image.
#         The default value is :obj:`0.3`.
#     max_scale : float
#         The maximum ratio between a cropped region and the original image.
#         The default value is :obj:`1`.
#     max_aspect_ratio : float
#         The maximum aspect ratio of cropped region.
#         The default value is :obj:`2`.
#     max_trial : int
#         Maximum number of trials for each constraint before exit no matter what.
#
#     Returns
#     -------
#     numpy.ndarray
#         Cropped bounding boxes with shape :obj:`(M, 4+)` where M <= N.
#     tuple
#         Tuple of length 4 as (x_offset, y_offset, new_width, new_height).
#
#     """
#     assert bbox.shape[1] >= 4, 'Each bbox at least 4 attribute,but got {}'.format(bbox.shape[1])
#     assert max_scale == 1, 'Required Max scale=1,but got {}'.format(max_scale)
#
#     candidates = []
#     mis, mas, mir, mar = min_scale, max_scale, min_aspect_ratio, max_aspect_ratio
#     sample_params = [
#         [1, 1, 1, 1],
#         [1, 1, mir, mar],
#         [mis, mas, 1, 1],
#         [mis, mas, mir, mar]]
#
#     # try to random crop
#     H, W = size
#     for i in range(4):
#         mis, mas, mir, mar = sample_params[i]
#         for _ in range(max_trial):
#             # randomly choose a scale and a ratio
#             s = random.uniform(mis, mas)
#             ar = random.uniform(
#                 max(mir, s ** 2),
#                 min(mar, 1 / (s ** 2)))
#             if W >= H * ar:
#                 crop_H = H * s
#                 crop_W = crop_H * ar
#             else:
#                 crop_W = W * s
#                 crop_H = crop_W / ar
#             crop_W, crop_H = int(crop_W), int(crop_H)
#             # randomly choose a area
#             crop_l = random.randrange(W - crop_W + 1)
#             crop_t = random.randrange(H - crop_H + 1)
#             crop_bb = np.array((crop_l, crop_t, crop_l + crop_W, crop_t + crop_H))
#             iob = bbox_iob(bbox, crop_bb[np.newaxis]).flatten()
#             iob = iob[iob > 0]
#
#             if len(iob) > bbox.shape[0] * 0.75 and iob.min() >= min_object_overlap - eps:
#                 if i != 3:
#                     candidates.append((crop_l, crop_t, crop_W, crop_H))
#                 else:
#                     candidates.extend([(crop_l, crop_t, crop_W, crop_H)] * 6)
#                 break
#     # random select one
#     while candidates:
#         crop = candidates.pop(random.randint(0, len(candidates)))
#         new_bbox = gbbox.crop(bbox, crop, allow_outside_center=False)
#         if new_bbox.size < 1:
#             continue
#         new_crop = (crop[0], crop[1], crop[2], crop[3])
#         return new_bbox, new_crop
#
#     min_len = int(min(H, W) * random.uniform(min_scale, max_scale))
#     crop_h, crop_w = min_len, min_len
#     for _ in range(max_trial):
#         crop_t = random.randrange(H - crop_h + 1)
#         crop_l = random.randrange(W - crop_w + 1)
#         crop = (crop_l, crop_t, crop_w, crop_h)
#         new_bbox = gbbox.crop(bbox, crop, allow_outside_center=False)
#         if new_bbox.size >= bbox.size * 0.5:
#             return new_bbox, crop
#
#     return bbox, (0, 0, W, H)
#
#
# def bbox_iob(bbox_a, bbox_b):
#     """Calculate Intersection-Over-Object(IOB) of two bounding boxes.
#     ! differenct between Fast R-CNN bbox_overlaps and gluon-cv bbox_iou
#
#     Parameters
#     ----------
#     bbox_a : numpy.ndarray, object bbox
#         An ndarray with shape :math:`(N, 4)`.
#     bbox_b : numpy.ndarray, crop bbox
#         An ndarray with shape :math:`(M, 4)`.
#
#     Returns
#     -------
#     numpy.ndarray
#         An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
#         bounding boxes in `bbox_a` and `bbox_b`.
#
#     """
#     # broadcast to each item [N,1,2]      [M,2]------>[1,M,2]
#     tl = np.maximum(bbox_a[:, np.newaxis:0:2], bbox_b[:, 0:2])
#     br = np.minimum(bbox_a[:, np.newaxis, 2:4], bbox_b[:, 2:4]) + 1
#
#     area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
#     area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + 1, axis=1)
#
#     return area_i / area_a[:, np.newaxis]