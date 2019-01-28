"""Train PyramidBox end2end"""
import os

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

import time
import argparse
import logging
import mxnet as mx
from mxnet import autograd, nd
import gluoncv as gcv
from gluoncv import utils as gutils
from gluoncv.data.batchify import Tuple, Stack, Pad

from mxnet import gluon
from pyrimidbox.nn import get_pyramidbox
from pyrimidbox.data import PyramidBoxTrainTransform, PyramidBoxValTransform
from pyrimidbox.data import WiderDetection, WiderFaceMetric, WiderFaceEvalMetric


def parse_args():
    parser = argparse.ArgumentParser(description='Train PyramidBox end2end')
    parser.add_argument('--network', type=str, default='VGG16',
                        help="Base network name which serves as feature extraction base")
    parser.add_argument('--use-bn', action='store_true',
                        help="Whether to use batchnorm layer in base model.")
    parser.add_argument('--data-shape', type=int, default=640,
                        help="Input data shape,only support 640 currently")
    parser.add_argument('--dataset', type=str, default='train',
                        help="Training dataset, Now support train,train,val.")
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training mini-batch size')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int, default=-1,
                        help="Number of data workers.Multi-thread to accelerate data loading.if your CPU and GPU are powerful.")
    parser.add_argument('--gpus', type=str, default='0,',
                        help="Training with GPUs, you can specify 1,2,3 for example.")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Training epochs.")
    parser.add_argument("--resume", type=str, default='',
                        help="Resume from previously saved parameters if not None."
                             "For example,you can resume from ./pyramidbox_xxx_212.params")
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate,default is 0.01.')
    parser.add_argument('--lr-decay', type=float, default=0.94,
                        help='Decay rate of learning rate. default is 0.94.')
    parser.add_argument('--lr-decay-epoch', type=str, default='80,160,200',
                        help='Epoches at which learning rate decay. default is 160,200.')
    parser.add_argument('--lr-warmup', type=str, default='',
                        help='warmup iterations to adjust learning rate, default is 0 for voc.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is  0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--grad-clip', type=float, default=2.0,
                        help='Gradient clip, default is 2.0')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='models/',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Saving parameters epoch interval,best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=5,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed')
    parser.add_argument('--match-high-thresh', type=float, default=0.35,
                        help='High threshold for anchor matching.')
    parser.add_argument('--match-low-thresh', type=float, default=0.1,
                        help='Low threshold for anchor matching.')
    parser.add_argument('--match-topk', type=int, default=6,
                        help='Topk for anchor matching.')
    args = parser.parse_args()
    args.lr_warmup = args.lr_warmup if args.lr_warmup else 1000
    return args


def get_dataset(dataset):
    # get train and valid dataset
    if dataset == 'train,val':
        dataset = ('train', 'val')
    else:
        assert dataset == 'train', "Invalid training dataset: {}".format(dataset)
    train_dataset = WiderDetection(root='/home/kevin/yuncong', splits=dataset)
    val_dataset = WiderDetection(root='/home/kevin/yuncong', splits='custom')
    val_metric = WiderFaceMetric(iou_thresh=0.5)
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, args):
    """Get dataloader: transform and batchify."""
    height, width = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, face_anchors, \
        _, _, head_anchors, \
        _, _, body_anchors = net(mx.nd.zeros((1, 3, height, width)))
    anchors = [face_anchors, head_anchors, body_anchors]
    # stack image,cls_target box_target
    train_batchify_fn = Tuple(Stack(),  # source img
                              Stack(), Stack(), Stack(),  # face_cls_targets,head_cls_targets,body_cls_targets
                              Stack(), Stack(), Stack())  # face_box_targets,head_box_targets,body_cls_targets
    # getdataloader
    train_loader = gluon.data.DataLoader(train_dataset.transform(
        PyramidBoxTrainTransform(width, height, anchors)),
        batch_size=batch_size, shuffle=True,
        batchify_fn=train_batchify_fn, num_workers=num_workers, last_batch='rollover')
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(PyramidBoxValTransform()),
        batch_size=batch_size, shuffle=False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader


def save_params(net, logger, best_map, current_map, maps, epoch, save_interval, prefix):
    current_map = float(current_map)
    model_path = '{:s}_{:03d}.params'.format(prefix, epoch)
    best_path = '{:s}_best.params'.format(prefix)
    # msg = '{:03d}:\t{:.4f}\t {:.6f} {:.6f} {:.6f}'.format(epoch, current_map, *maps)

    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters(best_path)
        with open(prefix + '_best_maps.log', 'a') as f:
            msg = '{:03d}:\t{:.4f}\t {:.6f} {:.6f} {:.6f}'.format(epoch, current_map, *maps)
            f.write(msg + '\n')

    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params').format(prefix, epoch, current_map))
        net.save_parameters(model_path)


def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    # net.input_reshape((1024, 1024))
    # net.collect_params().reset_ctx(ctx)
    eval_metric.reset()
    # set nms threshold and topk constraint
    # net.set_nms(nms_thresh=0.3, nms_topk=5000, post_nms=750)
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_scores = []
        gt_bboxes = []
        gt_lists = []
        for x, y in zip(data, label):
            # print('y shape',y.shape)
            _, scores, bboxes = net(x)
            det_scores.append(scores)
            det_bboxes.append(bboxes)
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_lists.append(y.slice_axis(axis=-1, begin=4, end=7))
        # update metric
        eval_metric.update(det_bboxes, det_scores, gt_bboxes, gt_lists)
    return eval_metric.get()


def get_lr_at_iter(alpha):
    return alpha


def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipline"""
    net.collect_params().reset_ctx(ctx)
    # training_patterns = '.*vgg'
    # net.collect_params(training_patterns).setattr('lr_mult', 0.1)
    trainer = gluon.Trainer(
        net.collect_params(),
        'sgd',
        {'clip_gradient': args.grad_clip,
         'learning_rate': args.lr,
         'momentum': args.momentum,
         'wd': args.wd,
         })
    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
    lr_warmup = float(args.lr_warmup)

    face_mbox_loss = gcv.loss.SSDMultiBoxLoss(rho=0.5,lambd=0.6)
    head_mbox_loss = gcv.loss.SSDMultiBoxLoss(rho=0.5,lambd=0.6)
    body_mbox_loss = gcv.loss.SSDMultiBoxLoss(rho=0.5,lambd=0.6)
    face_ce_metric = mx.metric.Loss('FaceCrossEntropy')
    face_smoothl1_metric = mx.metric.Loss('FaceSmoothL1')
    head_ce_metric = mx.metric.Loss('HeadCrossEntropy')
    head_smoothl1_metric = mx.metric.Loss('HeadSmoothL1')
    body_ce_metric = mx.metric.Loss('BodyCrossEntropy')
    body_smoothl1_metric = mx.metric.Loss('BodySmoothL1')
    metrics = [face_ce_metric, face_smoothl1_metric,
               head_ce_metric, head_smoothl1_metric,
               body_ce_metric, body_smoothl1_metric]

    # set up loger
    logger = logging.getLogger()  # formatter = logging.Formatter('[%(asctime)s] %(message)s')
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if os.path.exists(log_file_path) and args.start_epoch == 0:
        os.remove(log_file_path)
    fh = logging.FileHandler(log_file_path)

    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]

    total_batch = 0
    base_lr = trainer.learning_rate
    # names, maps = validate(net, val_data, ctx, eval_metric)

    for epoch in range(args.start_epoch, args.epochs + 1):
        # while lr_steps and epoch >= lr_steps[0]:
        #     new_lr = trainer.learning_rate * lr_decay
        #     lr_steps.pop(0)
        #     trainer.set_learning_rate(new_lr)
        #     logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        # every epoch learning rate decay
        if args.start_epoch != 0 or total_batch >= lr_warmup:
            new_lr = trainer.learning_rate * lr_decay
            # lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))

        face_ce_metric.reset()
        face_smoothl1_metric.reset()
        head_ce_metric.reset()
        head_smoothl1_metric.reset()
        body_ce_metric.reset()
        body_smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()

        for i, batch in enumerate(train_data):
            # if epoch == 0 and i <= lr_warmup:
            #     # adjust based on real percentage
            #     new_lr = base_lr * get_lr_at_iter((i + 1) / lr_warmup)
            #     if new_lr != trainer.learning_rate:
            #         if i % args.log_interval == 0:
            #             logger.info('[Epoch 0 Iteration {}] Set learning rate to {}'.format(i, new_lr))
            #         trainer.set_learning_rate(new_lr)
            if args.start_epoch == 0 and total_batch <= lr_warmup:
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter((total_batch + 1) / lr_warmup)
                if new_lr != trainer.learning_rate:
                    if i % args.log_interval == 0:
                        logger.info(
                            '[Epoch {} Iteration {}] Set learning rate to {}'.format(epoch, total_batch, new_lr))
                    trainer.set_learning_rate(new_lr)
            total_batch += 1
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            face_cls_target = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            head_cls_target = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            body_cls_target = gluon.utils.split_and_load(batch[3], ctx_list=ctx, batch_axis=0)

            face_box_target = gluon.utils.split_and_load(batch[4], ctx_list=ctx, batch_axis=0)
            head_box_target = gluon.utils.split_and_load(batch[5], ctx_list=ctx, batch_axis=0)
            body_box_target = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)

            with autograd.record():
                face_cls_preds = []
                face_box_preds = []
                head_cls_preds = []
                head_box_preds = []
                body_cls_preds = []
                body_box_preds = []

                for x in data:
                    face_cls_predict, face_box_predict, _, \
                    head_cls_predict, head_box_predict, _, \
                    body_cls_predict, body_box_predict, _ = net(x)
                    face_cls_preds.append(face_cls_predict)
                    face_box_preds.append(face_box_predict)
                    head_cls_preds.append(head_cls_predict)
                    head_box_preds.append(head_box_predict)
                    body_cls_preds.append(body_cls_predict)
                    body_box_preds.append(body_box_predict)
                # calculate the loss
                face_sum_loss, face_cls_loss, face_box_loss = face_mbox_loss(
                    face_cls_preds, face_box_preds, face_cls_target, face_box_target)

                head_sum_loss, head_cls_loss, head_box_loss = head_mbox_loss(
                    head_cls_preds, head_box_preds, head_cls_target, head_box_target)

                body_sum_loss, body_cls_loss, body_box_loss = body_mbox_loss(
                    body_cls_preds, body_box_preds, body_cls_target, body_box_target)

                # use 1:0.5:0.2 to backward loss
                # totalloss = [face_sum_loss,head_sum_loss,body_sum_loss]
                totalloss = [f + 0 * h + 0 * b for f, h, b in zip(face_sum_loss, head_sum_loss, body_sum_loss)]
                # totalloss = face_sum_loss+head_sum_loss
                autograd.backward(totalloss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            # logger training info
            face_ce_metric.update(0, [l * batch_size for l in face_cls_loss])
            face_smoothl1_metric.update(0, [l * batch_size for l in face_box_loss])
            head_ce_metric.update(0, [l * batch_size for l in head_cls_loss])
            head_smoothl1_metric.update(0, [l * batch_size for l in head_box_loss])
            body_ce_metric.update(0, [l * batch_size for l in body_cls_loss])
            body_smoothl1_metric.update(0, [l * batch_size for l in body_box_loss])

            # save_params(net, logger, [0], 0, None, total_batch, args.save_interval, args.save_prefix)

            if args.log_interval and not (i + 1) % args.log_interval:
                info = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
                # print(info)
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec {:s}'.format(
                    epoch, i, batch_size / (time.time() - btic), info))
            btic = time.time()
        info = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
        logger.info('[Epoch {}] Training cost: {:s}'.format(epoch, info))

        if args.val_interval and not (epoch + 1) % args.val_interval:
            # consider reduce the frequency of validation to save time
            vtic = time.time()
            names, maps = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{:7}MAP = {}'.format(k, v) for k, v in zip(names, maps)])
            logger.info('[Epoch {}] Validation: {:.3f}\n{}'.format(epoch, (time.time() - vtic), val_msg))
            current_map = sum(maps) / len(maps)
            # save_params(net, logger,best_map, current_map, maps, epoch, args.save_interval, args.save_prefix)
        else:
            current_map = 0.
            maps = None
        save_params(net, logger, best_map, current_map, maps, epoch, args.save_interval, args.save_prefix)


if __name__ == '__main__':
    args = parse_args()
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    net = get_pyramidbox(args.network, args.use_bn, pretrained=args.resume)
    network = args.network + ('_bn' if args.use_bn else '')
    args.save_prefix = os.path.join(args.save_prefix, network, 'pyramidbox')
    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset)
    train_data, val_data = get_dataloader(
        net, train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers, args)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args)
