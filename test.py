"""PramidBox test script"""
import os

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import argparse
import mxnet as mx
import gluoncv as gcv
from matplotlib import pyplot as plt

from pyrimidbox import get_pyramidbox
from pyrimidbox import PyramidBoxDetector
plt.switch_backend('agg')

def parse_args():
    parser = argparse.ArgumentParser(description='Test with pyrimidbox networks.')
    parser.add_argument('--network', '-n', type=str, default='VGG16',
                        help="Base network name")
    parser.add_argument('--use-bn', action='store_true',
                        help="Whether enable base model to use batch-norm layer.")
    parser.add_argument('--model', '-m', type=str, default='',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--image', type=str, default='tools/selfie.jpg')
    parser.add_argument('--gpu', type=int, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    args = parser.parse_args()
    return args


def get_detector(name, use_bn, model, ctx):
    net = get_pyramidbox(name, use_bn=use_bn, pretrained=model)
    net.input_reshape((6000, 2048))
    base = 1
    return PyramidBoxDetector(net, base, ctx)


if __name__ == '__main__':
    args = parse_args()
    # context
    ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
    detector = get_detector(args.network, args.use_bn, args.model, ctx)
    img = mx.image.imread(args.image)
    scores, bboxes = detector.detect(img)
    ax = gcv.utils.viz.plot_bbox(img, bboxes, thresh=0.3)
    plt.show()
