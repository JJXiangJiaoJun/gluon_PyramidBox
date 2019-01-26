"""Feature Pyramid Net"""
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn
from mxnet import nd
import mxnet as mx


class TopDownBlock(HybridBlock):
    """Feature Pyramid Net Top-down pathway and lateral connection
        ref:
    Parameters
    ----------
    out_plain : int
        Output channel of this topdown model.
    """

    def __init__(self, topdown_out_plain, **kwargs):
        super(TopDownBlock, self).__init__(**kwargs)
        self._out_plain = topdown_out_plain
        with self.name_scope():
            self.top_conv = nn.Conv2D(channels=topdown_out_plain, kernel_size=1)
            self.lateral_conv = nn.Conv2D(channels=topdown_out_plain, kernel_size=1)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, top, lateral):
        _, _, H, W = lateral.shape
        top = self.top_conv(top)
        lateral = self.lateral_conv(lateral)
        # upsample
        top = F.contrib.BilinearResize2D(top, width=W, height=H)
        # print('topshape=',top.shape)
        # if lateral.shape[2]!=H or lateral.shape[3]!=W:
        #     lateral = F.slice_like(data=lateral,shape_like=top,axes=(2,3)
        out = top + lateral
        return out


class LowLevelFeaturePyramidBlock(HybridBlock):
    """Low-Level Feature Pyramid Block use to connect high resolution feature map
       and contextual feature map.

    Parameters
    ----------
    topdown_out_plain : int
        Output channel of topdown Block model.
    smooth_out_plain : int
        Output channel of smooth Block model.
    """

    def __init__(self, topdown_out_plain, smooth_out_plain, **kwargs):
        super(LowLevelFeaturePyramidBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.topdown = TopDownBlock(topdown_out_plain)
            self.smooth = nn.Conv2D(smooth_out_plain, kernel_size=3, padding=1)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, top, lateral):
        topdown_out = self.topdown(top, lateral)
        smooth_out = self.smooth(topdown_out)
        return smooth_out


if __name__ == '__main__':
    topdownmodule = TopDownBlock(out_plain=60)
    topdownmodule.initialize(ctx=mx.gpu())
    top = nd.random.uniform(shape=(3, 3, 128, 128), ctx=mx.gpu())
    lateral = nd.random.uniform(shape=(3, 3, 256, 256), ctx=mx.gpu())
    out = topdownmodule(top, lateral)
    print(out.shape)
