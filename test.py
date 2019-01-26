from pyrimidbox import get_pyramidbox
from pyrimidbox.nn import VGG16
import mxnet as mx
import os
from mxnet import autograd
# os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']=0
if __name__ == '__main__':
    net = get_pyramidbox(VGG16, use_bn=True)
    net.collect_params().reset_ctx(mx.gpu())
    x = mx.nd.random.uniform(shape=(10, 3, 640, 640), ctx=mx.gpu())
    # with autograd.train_mode():
    #     res=net(x)
    res = net(x)
    # face_cls_predicts, face_box_predicts, face_anchors, \
    # head_cls_predicts, head_box_predicts, head_anchors, \
    # body_cls_predicts, body_box_predicts, body_anchors = res
    #
    # print('face_cls_predicts', face_cls_predicts.shape)
    # print('face_box_predicts', face_box_predicts.shape)
    # print('face_anchors', face_anchors.shape)
    #
    # print('head_cls_predicts', head_cls_predicts.shape)
    # print('head_box_predicts', head_box_predicts.shape)
    # print('head_anchors', head_anchors.shape)
    #
    # print('body_cls_predicts', body_cls_predicts.shape)
    # print('body_box_predicts', body_box_predicts.shape)
    # print('body_anchors', body_anchors.shape)
    ids, scores, bboxes =res
    print('body_cls_predicts', ids.shape)
    print('body_box_predicts', scores.shape)
    print('body_anchors', bboxes.shape)
