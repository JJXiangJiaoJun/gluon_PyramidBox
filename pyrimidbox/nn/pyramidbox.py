# pylint: disable=unused-import
"""PyramidBox base line model"""
from __future__ import absolute_import

import mxnet as mx
from mxnet import autograd
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn
import gluoncv as gcv
from .anchor import PyramidBoxAnchorGenerator
from .predictor import ContextSensitiveModule, ConvMaxInOutPredictor
from .fpn import LowLevelFeaturePyramidBlock
from ..data import YunCongDetection, WiderDetection
from .vgg import VGG16

_models = {
    'VGG16': VGG16}


class PyramidBox(HybridBlock):
    """PyramidBox: A Context-assisted Single Shot Face Detector. https://arxiv.org/pdf/1803.07737.pdf.

    Parameters
    ----------
    features : list of str or mxnet.gluon.HybridBlock
        Intermediate features to be extracted or a network with multi-output.
    base_size : int
        Base input size, it is speficied so SFD can support dynamic input shapes.
    sizes : iterable fo float
        Sizes of anchor boxes, this should be a list of floats, in incremental order.
        The length of `sizes` must be len(layers) + 1. For example, a two stage SFD
        model can have ``sizes = [30, 60, 90]``, and it converts to `[30, 60]` and
        `[60, 90]` for the two stages, respectively. For more details, please refer
        to original paper.
    ratios : iterable of list
        Aspect ratios of anchors in each output layer. Its length must be equals
        to the number of SFD output layers.
    steps : dict of str to list
        Step size of anchor boxes in each output layer.It should include 3 items.
        face,head,body
    classes : iterable of str
        Names of all categories.
    use_bn : bool
        Whether to use BatchNorm layer after each attached convolutional layer.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    stds : tuple of float, default is (0.1, 0.1, 0.2, 0.2)
        Std values to be divided/multiplied to box encoded values.
    nms_thresh : float, default is 0.45.
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    ctx : mx.Context
        Network context.


    """

    def __init__(self, features, base_size, sizes, ratios,
                 steps, classes, use_bn=False, pretrained=False,
                 stds=(0.1, 0.1, 0.2, 0.2), nms_thresh=0.35,
                 nms_topk=5000, post_nms=1500, ctx=mx.cpu(), **kwarg):
        super(PyramidBox, self).__init__(**kwarg)
        assert base_size == 640, "Currently only 640 is supported!"
        assert isinstance(steps, dict), "Must provide steps as dict str to list include face,head,body"
        # check Parameters
        num_layers = len(steps['face'])
        assert isinstance(sizes, list), "Must provide sizes as list or list of list"
        assert isinstance(ratios, list), "Must provide ratios as list or list or list"
        if not isinstance(ratios[0], (tuple, list)):
            ratios = [ratios] * num_layers  # propagate to all layers if use same ratio
        assert num_layers == len(sizes) == len(ratios), \
            'Mismatched (number of layers) vs (sizes) vs (ratios): {} ,{},{}'.format(
                num_layers, len(sizes), len(ratios))
        assert num_layers > 0, "Pyramidbox require at least one layer,suggest multiple"
        self.features = features(batch_norm=use_bn, pretrained=pretrained, ctx=ctx)
        # self.features = features
        self._num_layers = num_layers
        self.classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self.base_size = base_size
        self.im_size = [base_size, base_size]

        with self.name_scope():
            # low-level feature pyramid net
            self.conv5_lfpn0 = LowLevelFeaturePyramidBlock(512, 512)
            self.conv4_lfpn1 = LowLevelFeaturePyramidBlock(512, 512)
            self.conv3_lfpn2 = LowLevelFeaturePyramidBlock(256, 256)
            # context-sensetive modules
            self.conv3_context = ContextSensitiveModule(out_plain=256)
            self.conv4_context = ContextSensitiveModule(out_plain=256)
            self.conv5_context = ContextSensitiveModule(out_plain=256)
            self.convfc7_context = ContextSensitiveModule(out_plain=256)
            self.conv6_context = ContextSensitiveModule(out_plain=256)
            self.conv7_context = ContextSensitiveModule(out_plain=256)

            # lateral layer
            self.convfc7_lateral = nn.Conv2D(1024, kernel_size=1,
                                             weight_initializer=mx.init.Xavier(magnitude=2), bias_initializer='zeros')
            # self.convfc7_lateral = nn.Conv2D(1024, kernel_size=1,
            #                                  )
            self.conv6_lateral = nn.Conv2D(512, kernel_size=1,
                                           weight_initializer=mx.init.Xavier(magnitude=2), bias_initializer='zeros')
            # self.conv6_lateral = nn.Conv2D(512, kernel_size=1,
            #                                )
            self.conv7_lateral = nn.Conv2D(256, kernel_size=1,
                                           weight_initializer=mx.init.Xavier(magnitude=2), bias_initializer='zeros')
            # self.conv7_lateral = nn.Conv2D(256, kernel_size=1,
            #                                )

            # generate anchors
            self.face_cls_predictors = nn.HybridSequential()
            self.face_box_predictors = nn.HybridSequential()
            self.head_cls_predictors = nn.HybridSequential()
            self.head_box_predictors = nn.HybridSequential()
            self.body_cls_predictors = nn.HybridSequential()
            self.body_box_predictors = nn.HybridSequential()

            self.face_anchor_generators = nn.HybridSequential()
            self.head_anchor_generators = nn.HybridSequential()
            self.body_anchor_generators = nn.HybridSequential()

            alloc_size = [base_size // 4, base_size // 4]

            for i in range(num_layers):
                # pregenerate prior box
                face_step = steps['face'][i]
                face_size = sizes[i]
                face_ratio = ratios[i]
                face_anchor_generator = PyramidBoxAnchorGenerator(i, self.im_size, face_size, face_ratio, face_step,
                                                                  alloc_size)
                self.face_anchor_generators.add(face_anchor_generator)
                if i >= 1:
                    head_step = steps['head'][i - 1]
                    head_size = sizes[i - 1]
                    head_ratio = ratios[i - 1]
                    head_anchor_generator = PyramidBoxAnchorGenerator(i - 1, self.im_size, head_size, head_ratio,
                                                                      head_step, alloc_size)
                    self.head_anchor_generators.add(head_anchor_generator)
                if i >= 2:
                    body_step = steps['body'][i - 2]
                    body_size = sizes[i - 2]
                    body_ratio = ratios[i - 2]
                    body_anchor_generator = PyramidBoxAnchorGenerator(i - 2, self.im_size, body_size, body_ratio,
                                                                      body_step, alloc_size)
                    self.body_anchor_generators.add(body_anchor_generator)
                # pre-compute larger than 16x16 anchor map
                alloc_size = [max(sz // 2, 32) for sz in alloc_size]
                # head_alloc_size = [max(sz // 2, 16) for sz in head_alloc_size]
                # body_alloc_size = [max(sz // 2, 16) for sz in body_alloc_size]
                # cls_predictor & box_predictor
                num_anchors = face_anchor_generator.num_depth
                assert num_anchors == 1, "Currently only support face number anchors=1"
                if i == 0:
                    face_cls_predictor = ConvMaxInOutPredictor(num_channel=num_anchors * 4, max_out=True)
                    self.face_cls_predictors.add(face_cls_predictor)
                else:
                    face_cls_predictor = ConvMaxInOutPredictor(num_channel=num_anchors * 4, max_out=False)
                    self.face_cls_predictors.add(face_cls_predictor)
                face_box_predictor = nn.Conv2D(num_anchors * 4, kernel_size=3, strides=1, padding=1,
                                               weight_initializer=mx.init.Xavier(magnitude=2), bias_initializer='zeros')
                # face_box_predictor = nn.Conv2D(num_anchors * 4, kernel_size=3, strides=1, padding=1,
                #                                )
                self.face_box_predictors.add(face_box_predictor)
                if i >= 1:
                    num_anchors = head_anchor_generator.num_depth
                    assert num_anchors == 1, "Currently only support head number anchors=1"
                    head_cls_predictor = nn.Conv2D(2 * num_anchors, kernel_size=3, strides=1, padding=1,
                                                   weight_initializer=mx.init.Xavier(magnitude=2),
                                                   bias_initializer='zeros')
                    # head_cls_predictor = nn.Conv2D(2 * num_anchors, kernel_size=3, strides=1, padding=1,
                    #                                )

                    self.head_cls_predictors.add(head_cls_predictor)
                    head_box_predictor = nn.Conv2D(4 * num_anchors, kernel_size=3, strides=1, padding=1,
                                                   weight_initializer=mx.init.Xavier(magnitude=2),
                                                   bias_initializer='zeros')
                    # head_box_predictor = nn.Conv2D(4 * num_anchors, kernel_size=3, strides=1, padding=1,
                    #                                )

                    self.head_box_predictors.add(head_box_predictor)
                if i >= 2:
                    num_anchors = body_anchor_generator.num_depth
                    assert num_anchors == 1, "Currently only support body number anchors=1"
                    body_cls_predictor = nn.Conv2D(2 * num_anchors, kernel_size=3, strides=1, padding=1,
                                                   weight_initializer=mx.init.Xavier(magnitude=2),
                                                   bias_initializer='zeros')
                    # body_cls_predictor = nn.Conv2D(2 * num_anchors, kernel_size=3, strides=1, padding=1,
                    #                                )
                    self.body_cls_predictors.add(body_cls_predictor)
                    body_box_predictor = nn.Conv2D(4 * num_anchors, kernel_size=3, strides=1, padding=1,
                                                   weight_initializer=mx.init.Xavier(magnitude=2),
                                                   bias_initializer='zeros')
                    # body_box_predictor = nn.Conv2D(4 * num_anchors, kernel_size=3, strides=1, padding=1,
                    #                                )
                    self.body_box_predictors.add(body_box_predictor)

            self.bbox_decoder = gcv.nn.coder.NormalizedBoxCenterDecoder(stds)
            self.cls_decoder = gcv.nn.coder.MultiPerClassDecoder(len(self.classes) + 1, thresh=0.01)

    def input_reshape(self, im_size=(1024, 1024)):
        assert min(im_size) >= self.base_size
        self.im_size = im_size
        alloc_size = [sz // 4 for sz in im_size]
        for fag, hag, bag in zip(self.face_anchor_generators, self.head_anchor_generators, self.body_anchor_generators):
            fag.reset_anchors(alloc_size)
            hag.reset_anchors(alloc_size)
            bag.reset_anchors(alloc_size)
            alloc_size = [max(sz // 2, 32) for sz in alloc_size]

    @property
    def num_classes(self):
        """Return number of foreground classes.

        Returns
        -------
        int
            Number of foreground classes

        """
        return len(self.classes)

    def set_nms(self, nms_thresh=0.3, nms_topk=5000, post_nms=1500):
        """Set non-maximum suppression parameters.

                Parameters
                ----------
                nms_thresh : float, default is 0.3.
                    Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
                nms_topk : int, default is 5000
                    Apply NMS to top k detection results, use -1 to disable so that every Detection
                     result is used in NMS.
                post_nms : int, default is 750
                    Only return top `post_nms` detection results, the rest is discarded. The number is
                    based on COCO dataset which has maximum 100 objects per image. You can adjust this
                    number if expecting more objects. You can use -1 to return all detections.

                Returns
                -------
                None

                """
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        # generate features
        conv3, conv4, conv5, conv_fc7, conv6, conv7 = self.features(x)
        conv_fc7 = self.convfc7_lateral(conv_fc7)
        conv6 = self.conv6_lateral(conv6)
        conv7 = self.conv7_lateral(conv7)

        # build up Pyramid feature network
        output_features = list()
        lfpn0 = self.conv5_lfpn0(conv_fc7, conv5)
        lfpn1 = self.conv4_lfpn1(lfpn0, conv4)
        lfpn2 = self.conv3_lfpn2(lfpn1, conv3)

        lfpn2 = self.conv3_context(lfpn2)
        lfpn1 = self.conv4_context(lfpn1)
        lfpn0 = self.conv5_context(lfpn0)
        conv_fc7 = self.convfc7_context(conv_fc7)
        conv6 = self.conv6_context(conv6)
        conv7 = self.conv7_context(conv7)

        output_features.append(lfpn2)
        output_features.append(lfpn1)
        output_features.append(lfpn0)
        output_features.append(conv_fc7)
        output_features.append(conv6)
        output_features.append(conv7)
        # contextual sensitive predictor

        face_cls_predicts = [F.flatten(F.transpose(fcp(feat), (0, 2, 3, 1)))
                             for feat, fcp in zip(output_features, self.face_cls_predictors)]
        face_box_predicts = [F.flatten(F.transpose(fbp(feat), (0, 2, 3, 1)))
                             for feat, fbp in zip(output_features, self.face_box_predictors)]
        face_anchors = [F.reshape(fag(feat), shape=(1, -1))
                        for feat, fag in zip(output_features, self.face_anchor_generators)]

        face_cls_predicts = F.concat(*face_cls_predicts, dim=1).reshape((0, -1, self.num_classes + 1))
        face_box_predicts = F.concat(*face_box_predicts, dim=1).reshape((0, -1, 4))
        face_anchors = F.concat(*face_anchors, dim=1).reshape((1, -1, 4))

        if autograd.is_training():
            head_cls_predicts = [F.flatten(F.transpose(hcp(feat), (0, 2, 3, 1)))
                                 for feat, hcp in zip(output_features[1:], self.head_cls_predictors)]
            head_box_predicts = [F.flatten(F.transpose(hbp(feat), (0, 2, 3, 1)))
                                 for feat, hbp in zip(output_features[1:], self.head_box_predictors)]
            head_anchors = [F.reshape(hag(feat), shape=(1, -1))
                            for feat, hag in zip(output_features[1:], self.head_anchor_generators)]
            head_cls_predicts = F.concat(*head_cls_predicts, dim=1).reshape((0, -1, self.num_classes + 1))
            head_box_predicts = F.concat(*head_box_predicts, dim=1).reshape((0, -1, 4))
            head_anchors = F.concat(*head_anchors, dim=1).reshape((1, -1, 4))

            body_cls_predicts = [F.flatten(F.transpose(bcp(feat), (0, 2, 3, 1)))
                                 for feat, bcp in zip(output_features[2:], self.body_cls_predictors)]
            body_box_predicts = [F.flatten(F.transpose(bbp(feat), (0, 2, 3, 1)))
                                 for feat, bbp in zip(output_features[2:], self.body_box_predictors)]
            body_anchors = [F.reshape(bag(feat), shape=(1, -1))
                            for feat, bag in zip(output_features[2:], self.body_anchor_generators)]

            body_cls_predicts = F.concat(*body_cls_predicts, dim=1).reshape((0, -1, self.num_classes + 1))
            body_box_predicts = F.concat(*body_box_predicts, dim=1).reshape((0, -1, 4))
            body_anchors = F.concat(*body_anchors, dim=1).reshape((1, -1, 4))

            return [face_cls_predicts, face_box_predicts, face_anchors,
                    head_cls_predicts, head_box_predicts, head_anchors,
                    body_cls_predicts, body_box_predicts, body_anchors]
        # @TODO: Test Phase need to nms .....
        bboxes = self.bbox_decoder(face_box_predicts, face_anchors)
        cls_ids, scores = self.cls_decoder(F.softmax(face_cls_predicts, axis=-1))
        results = []
        for i in range(self.num_classes):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i + 1)
            score = scores.slice_axis(axis=-1, begin=i, end=i + 1)
            # per class results
            per_result = F.concat(*[cls_id, score, bboxes], dim=-1)
            results.append(per_result)
        result = F.concat(*results, dim=1)
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.01,
                id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = F.slice_axis(result, axis=2, begin=0, end=1)
        scores = F.slice_axis(result, axis=2, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=2, begin=2, end=6)
        return ids, scores, bboxes


sizes = [16, 32, 64, 128, 256, 512]
ratios = [1]
steps = dict({
    'face': [4, 8, 16, 32, 64, 128],
    'head': [8, 16, 32, 64, 128, 128],
    'body': [16, 32, 64, 128, 128, 128]})


def get_pyramidbox(features, use_bn=False, pretrained=False, **kwargs):
    """Get pyramidbox model

    Parameters
    ----------
    features : list of str or mxnet.gluon.HybridBlock
        Intermediate features to be extracted or a network with multi-output.
    use_bn : bool
        If `True`, will use BatchNorm layers.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    """
    pretrained_base = False if pretrained else True
    features = _models[features]
    # features_extractor = _models[features](batch_norm=use_bn, pretrained=False)
    net = PyramidBox(features, base_size=640, sizes=sizes, ratios=ratios, steps=steps,
                     classes=WiderDetection.CLASSES, use_bn=use_bn,
                     pretrained=pretrained_base, **kwargs)
    # net.collect_params().initialize(mx.init.Xavier())
    # net.features = _models[features](batch_norm=use_bn, pretrained=pretrained_base)
    if pretrained:
        assert isinstance(pretrained, str), "pretrained represents path to pretrained model."
        net.load_parameters(pretrained)
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()
    return net

# if __name__ == '__main__':
#     net = get_pyramidbox(VGG16, use_bn=True)
#     net.collect_params().reset_ctx(mx.gpu())
#     x = mx.nd.random.uniform(shape=(1, 3, 640, 640), ctx=mx.gpu())
#     res = net(x)
#
#     face_cls_predicts, face_box_predicts, face_anchors, \
#     head_cls_predicts, head_box_predicts, head_anchors, \
#     body_cls_predicts, body_box_predicts, body_anchors = net(x)
#
#     print('face_cls_predicts', face_cls_predicts.shape)
#     print('face_box_predicts', face_box_predicts.shape)
#     print('face_anchors', face_anchors.shape)
#
#     print('head_cls_predicts', head_cls_predicts.shape)
#     print('head_box_predicts', head_box_predicts.shape)
#     print('head_anchors', head_anchors.shape)
#
#     print('body_cls_predicts', body_cls_predicts.shape)
#     print('body_box_predicts', body_box_predicts.shape)
#     print('body_anchors', body_anchors.shape)
