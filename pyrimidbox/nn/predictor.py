"""Context-sensitive prediction Module"""
from __future__ import division
import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn


class ConvMaxInOutPredictor(HybridBlock):
    """Convolutional Max-in-out background classification predictor for PyramidBox
        It is useful to reduce false positive and improve recall rate.
        ref:

    Parameters
    ---------
    num_channel : int
        Number of conv channels.
    max_in : bool
        If max_int is true, prediction will be cp=1,cn=3 to reduce positive,
        else prediction will be cp=3,cn=1 to improve recall.More details show
        in the paper.
    kernel : tuple of (int, int), default (3, 3)
        Conv kernel size as (H, W).
    pad : tuple of (int, int), default (1, 1)
        Conv padding size as (H, W).
    stride : tuple of (int, int), default (1, 1)
        Conv stride size as (H, W).
    activation : str, optional
        Optional activation after conv, e.g. 'relu'.
    use_bias : bool
        Use bias in convolution. It is not necessary if BatchNorm is followed.
    """

    def __init__(self, num_channel=4, max_in=False, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                 activation='relu', use_bias=True, **kwargs):
        super(ConvMaxInOutPredictor, self).__init__(**kwargs)
        assert num_channel == 4, "Required num_channel = 4 but got {}".format(num_channel)
        self.num_channel = num_channel
        self.max_in = max_in
        with self.name_scope():
            self.predictor = nn.Conv2D(num_channel, kernel_size=kernel, strides=stride,
                                       padding=pad, activation=activation, use_bias=use_bias,
                                       weight_initializer=mx.init.Xavier(magnitude=2), bias_initializer='zeros')

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        x = self.predictor(x)
        if self.max_in:
            fc = F.slice_axis(x, axis=1, begin=0, end=1)
            bg = F.slice_axis(x, axis=1, begin=1, end=None)
            bg = F.max_axis(bg, axis=1, keepdims=True)
        else:
            fc = F.slice_axis(x, axis=1, begin=0, end=3)
            bg = F.slice_axis(x, axis=1, begin=3, end=None)
            fc = F.max_axis(fc, axis=1, keepdims=True)

        return F.concat(fc, bg, dim=1)


class ContextSensitiveModule(HybridBlock):
    """Context-sensitive prediction Module reference to SSH
    Parameters
    ----------
    out_plains : int
        Output channel of SSH context detection module.
    depth : int
        Number depth of the anchor.
    max_in : bool
        If max_int is true, prediction will be cp=1,cn=3 to reduce positive,
        else prediction will be cp=3,cn=1 to improve recall.More details show
        in the paper.

    """

    def __init__(self, out_plain=256, depth=1, max_in=False, **kwargs):
        super(ContextSensitiveModule, self).__init__(**kwargs)
        # pylint: disable=arguments-differ
        assert depth == 1, 'Currently only support depth=1'
        self._out_plain = out_plain
        self._depth = depth
        self._max_in = max_in
        with self.name_scope():
            self.SSH_Conv_1 = nn.Conv2D(channels=out_plain, kernel_size=3, strides=1,
                                        padding=1, activation='relu')
            self.SSH_Conv_2 = nn.Conv2D(channels=out_plain // 2, kernel_size=3, strides=1,
                                        padding=1, activation='relu')
            self.SSH_Conv_2_1 = nn.Conv2D(channels=out_plain // 2, kernel_size=3, strides=1,
                                          padding=1, activation='relu')
            self.SSH_Conv_2_2_1 = nn.Conv2D(channels=out_plain // 2, kernel_size=3, strides=1,
                                            padding=1, activation='relu')
            self.SSH_Conv_2_2_2 = nn.Conv2D(channels=out_plain // 2, kernel_size=3, strides=1,
                                            padding=1, activation='relu')

            # self.cls_predictor = ConvMaxInOutPredictor(num_channel=4 * depth, max_in=max_in)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        # output X channels
        x_conv_1 = self.SSH_Conv_1(x)

        x_conv_2 = self.SSH_Conv_2(x)
        x_conv_2_1 = self.SSH_Conv_2_1(x_conv_2)
        x_conv_2_2 = self.SSH_Conv_2_2_2(self.SSH_Conv_2_2_1(x_conv_2))
        x_conv_2 = F.concat(x_conv_2_1, x_conv_2_2, dim=1)

        out = F.concat(x_conv_1, x_conv_2, dim=1)

        # cls_preds = self.cls_predictor(out)

        return out


if __name__ == '__main__':
    predictor = ContextSensitiveModule()
    predictor.initialize(ctx=mx.gpu())
    feature = mx.nd.random.uniform(shape=(2, 3, 32, 32), ctx=mx.gpu())
    cls_preds, box_preds = predictor(feature)
    print(cls_preds[0])
    print(box_preds[0])
