"""yuncong face detection dataset"""

from __future__ import division
from __future__ import absolute_import

import os
import logging
import numpy as np
import mxnet as mx
from mxnet.gluon.data import dataset
from gluoncv.utils.bbox import bbox_xywh_to_xyxy
from easydict import EasyDict as edict
from scipy import io
import cv2

__all__ = ['YunCongDetection', 'WiderDetection']


class WiderDetection(dataset.Dataset):
    """Wider Face detection Dataset.

    Parameters
    ----------
    root : str, default 'datasets/widerface'
        Path to folder storing the dataset.
    splits : list of tuples, default ('train', 'val')
        Subset name
        Candidates can be: 'train', 'custom', val', ('train', 'val')
        After 2017-03-31 new val ground truth mat use same rounding as txt annotation.
    transform : callable, defaut None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for face detection should take label into consideration,
        because any geometric modification will require label to be modified.
    skip_empty : bool, default is True
        Whether skip images with no valid face. This should be `True` in training, otherwise
        it will cause undefined behavior.
    """
    CLASSES = ('face',)

    def __init__(self, root='widerface', splits=('train', 'val'),
                 transform=None, skip_empty=True):
        if not os.path.isdir(root):
            helper_msg = "{} is not a valid dir. Did you forget download and parpare widerface datasets.".format(root)
            raise OSError(helper_msg)
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._skip_empty = skip_empty
        if isinstance(splits, mx.base.string_types):
            splits = [splits]
        self._splits = splits
        self._items, self._labels = self._loadtxt()

    def __str__(self):
        detail = ','.join([str(s) for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    def _ignore_face(self, attr):
        return attr.height < 8

    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_path = self._items[idx]
        label = self._labels[idx]
        # img = mx.image.imread(img_path)
        img = cv2.imread(img_path)
        if self._transform is not None:
            return self._transform(img, label)
        else:
            return img, label

    def _loadtxt(self):
        """Load all annotations into memory."""
        logging.debug("Loading %s annotations into memory...", str(self))

        items = []
        labels = []
        for name in self._splits:
            img_root = os.path.join(self._root, 'WIDER_{}', 'images').format(name)
            anno_txt = os.path.join(self._root, 'wider_face_split',
                                    'wider_face_{}_bbx_gt.txt').format(name)
            with open(anno_txt, 'r') as f:
                while True:
                    img_path = f.readline().strip()
                    if img_path == '':
                        break
                    num = int(f.readline().strip())
                    label = []
                    for i in range(num):
                        annos = [int(a) for a in f.readline().strip().split()]
                        annos[2] += 1  # inter-w to real-w
                        annos[3] += 1  # inter-h to real-h
                        if 'train' in self._splits:
                            attr = edict()
                            attr.width = annos[2]
                            attr.height = annos[3]
                            attr.blur = annos[4]
                            attr.expression = annos[5]
                            attr.illumination = annos[6]
                            attr.invalid = annos[7]
                            attr.occlusion = annos[8]
                            attr.pose = annos[9]
                            if self._ignore_face(attr):
                                continue
                        if len(annos) == 7:
                            # annos = bbox_xywh_to_xyxy(map(float, annos[:4])) + tuple(annos[4:])
                            annos = bbox_xywh_to_xyxy(([float(item) for item in annos[:4]])) + tuple(annos[4:])
                        else:
                            # annos = bbox_xywh_to_xyxy(map(float, annos[:4]))
                            annos = bbox_xywh_to_xyxy(([float(item) for item in annos[:4]]))
                        label.append(annos)
                    if len(label) > 0 or not self._skip_empty:
                        items.append(os.path.join(img_root, img_path))
                        labels.append(np.array(label, dtype=np.float32))
        return items, labels


class YunCongDetection(dataset.Dataset):
    """ YunCong Face detection Dataset
    Parameters
    ----------
    root : str, default 'datasets/widerface'
        Path to folder storing the dataset.
    splits : list of tuples, default ('train')
        Subset name
        Candidates can be: 'train', 'custom', val', ('train', 'val')
        After 2017-03-31 new val ground truth mat use same rounding as txt annotation.
    transform : callable, defaut None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for face detection should take label into consideration,
        because any geometric modification will require label to be modified.
    skip_empty : bool, default is True
        Whether skip images with no valid face. This should be `True` in training, otherwise
        it will cause undefined behavior.
    dataset : str,defalut is 'our'
        Choose different dataset.
        Candidates can be : 'Mall','Our','Part_A','Part_B','UCSD'
    """
    CLASSES = ('face',)

    def __init__(self, root='yuncong_data', splits=('train',),
                 transform=None, skip_empty=True, dataset='our'):
        if not os.path.isdir(root):
            help_msg = "{} is not a valid dir.".format(root)
            raise OSError(help_msg)
        self._root = root
        self._transform = transform
        self._skip_empty = skip_empty
        if isinstance(splits, mx.base.string_types):
            splits = [splits]
        self._splits = splits
        self._dataset = dataset
        self._items, self._labels = self._load_txt()

    def __len__(self):
        return len(self._items)

    def __str__(self):
        detail = ','.join([str(s) for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def classes(self):
        return type(self).CLASSES

    def __getitem__(self, idx):
        img_path = self._items[idx]
        label = self._labels[idx]
        img = mx.image.imread(img_path)
        if self._transform is not None:
            return self._transform(img, label)
        else:
            return img, label

    def _load_txt(self):
        """load all annotations into memory"""
        logging.debug('Loading %s annotations into memory....', str(self))

        items = []
        labels = []

        data_root = os.path.join(self._root, self._dataset)
        for name in self._splits:
            anno_txt = os.path.join(data_root, 'yuncong_face_split',
                                    "{}_{}.txt").format(self._dataset, name)
            with open(anno_txt, 'r') as f:
                while True:
                    record = f.readline().strip().split()
                    if len(record) == 0:
                        break
                    imgpath = record[0].strip()
                    num = int(record[1].strip())
                    label = []
                    for i in range(num):
                        bbox = record[2 + i * 5:2 + (i + 1) * 5]
                        bbox = [float(x) for x in bbox]
                        bbox = bbox_xywh_to_xyxy(bbox[1:])
                        label.append(bbox)

                    if len(label) > 0 or not self._skip_empty:
                        items.append(os.path.join(data_root, imgpath))
                        labels.append(np.array(label, dtype=np.float32))
        return items, labels
