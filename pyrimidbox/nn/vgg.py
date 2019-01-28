"""VGG, implemented in Gluon"""
from __future__ import division

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

__all__ = ['VGG16', 'VGG19']


class VGGBase(gluon.HybridBlock):
    """
   VGG multi layer base network. You must inherit from it to define
    how the features are computed.

    Parameters
    ----------
    layers : list of int
        Number of layer for vgg base network.
    filters : list of int
        Number of convolution filters for each layer.
    batch_norm : bool, default is False
        If `True`, will use BatchNorm layers.
    
    """

    def __init__(self, layers, filters, batch_norm, **kwargs):
        super(VGGBase, self).__init__(prefix='vgg', **kwargs)
        self.init = dict({
            'weight_initializer': mx.init.Xavier(),
            'bias_initializer': 'zeros'})
        self.batch_norm = batch_norm
        assert len(layers) == len(filters), 'len(layers) must equals to len(filters)'
        self._layers = layers
        self._filters = filters

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features_')
            for i, config in enumerate(zip(layers, filters)):
                num_layer = config[0]
                out_planes = config[1]
                self.features.add(self._make_layers(i, num_layer, out_planes))
            # use convolution instead of dense layers

            features = nn.HybridSequential(prefix='fc_')
            with features.name_scope():
                features.add(nn.Conv2D(1024, kernel_size=3, padding=1, strides=1, **self.init))  # fc6
                if batch_norm:
                    features.add(nn.BatchNorm())
                features.add(nn.Activation('relu'))
                features.add(nn.Conv2D(1024, kernel_size=1, **self.init))  # fc7
                if batch_norm:
                    features.add(nn.BatchNorm())
                features.add(nn.Activation('relu'))
            self.features.add(features)

    def _make_layers(self, stage_index, num_layer, out_planes):
        layer = nn.HybridSequential(prefix='layer{}_'.format(stage_index))
        with layer.name_scope():
            for _ in range(num_layer):
                layer.add(nn.Conv2D(out_planes, kernel_size=3, strides=1, padding=1, **self.init))
                if self.batch_norm:
                    layer.add(nn.BatchNorm())
                layer.add(nn.Activation('relu'))
        return layer

    def hybrid_forward(self, F, x, init_scale):
        raise NotImplementedError

    def import_params(self, filename, ctx):
        """
        Load Parameters from gluoncv VGG.
        
        Parameters
        ----------
        filename : str 
            Path that store the paramters.
        ctx : mx.Context()
            mx.cpu() or mx.gpu()
        """
        print("import base model params from {}".format(filename))
        loaded = mx.nd.load(filename)
        params = self._collect_params_with_prefix()
        i = 0
        for k, l in enumerate(self._layers):
            j = 0
            for _ in range(l):
                # conv param
                for suffix in ['weight', 'bias']:
                    key_i = '.'.join(['features', str(i), suffix])
                    key_j = '.'.join(['features', str(k), str(j), suffix])

                    assert key_i in loaded, "Params '{}' missing in {}".format(key_i, loaded.keys())
                    assert key_j in params, "Params '{}' missing in {}".format(key_j, params.keys())

                    params[key_j]._load_init(loaded[key_i], ctx)

                i += 1
                j += 1

                if self.batch_norm:
                    # batch norm param
                    for suffix in ['beta', 'gamma', 'running_mean', 'running_var']:
                        key_i = '.'.join(('features', str(i), suffix))
                        key_j = '.'.join(('features', str(k), str(j), suffix))
                        assert key_i in loaded, "Params '{}' missing in {}".format(key_i, loaded.keys())
                        assert key_j in params, "Params '{}' missing in {}".format(key_j, params.keys())
                        params[key_j]._load_init(loaded[key_i], ctx)
                    i += 1
                    j += 1
                # relu
                i += 1
                j += 1
            # pooling
            i += 1
            # j += 1

        # stage 5
        params['features.5.0.weight']._load_init(
            loaded['features.%d.weight' % i].reshape(4096, 512, 7, 7)[:1024, :, 2:5, 2:5], ctx)
        params['features.5.0.bias']._load_init(
            loaded['features.%d.bias' % i][:1024], ctx)
        i += 2
        j = 3 if self.batch_norm else 2
        params['features.5.%d.weight' % j]._load_init(
            loaded['features.%d.weight' % i][:1024, :1024].reshape(1024, 1024, 1, 1), ctx)
        params['features.5.%d.bias' % j]._load_init(loaded['features.%d.bias' % i][:1024], ctx)


class VGGExtractor(VGGBase):
    """
    VGG multi layer feature extractor which produces multiple output feature maps
    
    Parameters
    ------------
    layers : list of int 
        Number of layer for vgg base network.
    filters : list of int 
        Number of convolution filters for each layer.
    extras : dict of str to list
        Extra layers configurations
    batch_norm : bool
        if `True`,will use BatchNorm layers
        
        
    """

    def __init__(self, layers, filters, extras,
                 batch_norm=False, **kwargs):
        super(VGGExtractor, self).__init__(layers, filters, batch_norm, **kwargs)
        with self.name_scope():
            self.extras_feature = nn.HybridSequential()
            for i, config in enumerate(extras['conv']):
                ex = nn.HybridSequential(prefix='extra%d_' % (i))
                with ex.name_scope():
                    for f, k, s, p in config:
                        ex.add(nn.Conv2D(f, k, s, p, **self.init))
                        if batch_norm:
                            ex.add(nn.BatchNorm())
                            ex.add(nn.Activation('relu'))

                self.extras_feature.add(ex)

    def hybrid_forward(self, F, x):
        assert len(self.features) == 6
        outputs = []
        """VGG feature extractor"""
        for layer in self.features[:2]:
            x = layer(x)
            x = F.Pooling(x, pool_type='max', kernel=(2, 2), stride=(2, 2),
                          pooling_convention='full')

        for layer in self.features[2:5]:
            x = layer(x)
            outputs.append(x)
            x = F.Pooling(x, pool_type='max', kernel=(2, 2), stride=(2, 2),
                          pooling_convention='full')
        x = self.features[5](x)
        outputs.append(x)
        """extra convolution feature extractor"""
        for layer in self.extras_feature:
            x = layer(x)
            outputs.append(x)
        return outputs


vgg_spec = {
    11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
    13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
    16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
    19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])
}

extra_spec = {
    'conv': [((256, 1, 1, 0), (512, 3, 2, 1)),  # conv6_
             ((128, 1, 1, 0), (256, 3, 2, 1))],  # conv7_
    # 'normalize': (10, 8, 5),
}


def get_vgg_extractor(num_layers, pretrained=False, ctx=mx.cpu(),
                      root='/home/kevin/vgg_params', **kwargs):
    """Get VGG feature extractor networks
    
    Parameters
    ----------
    num_layers : int
        VGG types,can be 11,13,16,19
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : mx.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.
        
    Returns
    -------
    mxnet.gluon.HybridBlock
        The returned network.    

    """
    layers, filters = vgg_spec[num_layers]
    net = VGGExtractor(layers, filters, extra_spec, **kwargs)
    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        batch_norm_suffix = '_bn' if kwargs.get('batch_norm') else ''
        net.initialize(ctx=ctx)
        assert num_layers >= 16, "current import_params only support vgg 16 or 19, but got {}".format(num_layers)
        net.import_params(get_model_file('vgg%d%s' % (num_layers, batch_norm_suffix),
                                         tag=pretrained, root=root), ctx=ctx)

    return net


def VGG16(**kwargs):
    """Get VGG16 feature extractor network"""
    return get_vgg_extractor(16, **kwargs)


def VGG19(**kwargs):
    """Get VGG19 feature extractor network"""
    return get_vgg_extractor(19, **kwargs)

# if __name__ == '__main__':
#     net = VGG16(batch_norm=True,pretrained=True,root='~/vgg_params')
#     # print(net)
#     net.collect_params().reset_ctx(mx.gpu())
#     # net.initialize(ctx=mx.gpu())
#     feat = mx.nd.random.uniform(shape=(1,3,640,640),ctx=mx.gpu())
#     conv3, conv4, conv5, conv_fc7, conv6, conv7 = net(feat)
#     print('conv3 ',conv3.shape)
#     print('conv4 ',conv4.shape)
#     print('conv5 ',conv5.shape)
#     print('conv_fc7',conv_fc7.shape)
#     print('conv6 ',conv6.shape)
#     print('conv7 ',conv7.shape)

# print('generate vgg16')
# net.import_params('test.params', mx.gpu())
# print(net.collect_params())
