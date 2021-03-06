# PyramidBox: A Context-assisted Single Shot Face Detector

## Introduction
A MXNet implementation of PyramiBbox:A Context-assisted Single Shot Face Detector.
If you want to learn more details,please refer to the original paper.
```
@inproceedings{Tang2018PyramidBoxAC,
  title={PyramidBox: A Context-Assisted Single Shot Face Detector},
  author={Xu Tang and Daniel K. Du and Zeqiang He and Jingtuo Liu},
  booktitle={ECCV},
  year={2018}
}
```
I train PyramidBox with WIDER FACE dataset,results are as follows:


|&emsp;&emsp;|Easy mAP|Medium mAP|Hard mAP|
|---|---|---|---|
|paper|**96.1**|**95.0**|**88.9**|
|this repo|**92.5**|**90.8**|**83.3**|




I think mainly reasons that this repo can not get the same precision as paper as follows:
* I use batch size 4 because of memory limitations,which is 16 in the paper
* some parameters are not metioned in the paper


Here are several examples of succesful detection outputs:
![](./pictures/detection1.png)
![](./pictures/detection2.png)
![](./pictures/detection4.png)
## Details
I implement following structures metioned in the paper:
- [x] Low-Level FPN
- [x] max-in-out layer
- [x] PyramidAnchors
- [x] Context-sensitive Prediction Module
- [x] Data-anchor sampling
- [x] Learning rate warmup and cosine decay   
## Dependencies
* Python 3.6.4
* [MXNet](https://github.com/apache/incubator-mxnet) 1.3.1
* [gluon-cv](https://github.com/dmlc/gluon-cv) 0.4.1

## Implement Details
* Ubuntu 16.04 LTS
* CUDA 9.0
* cuDNN 7.0.5

## Preparation
```bash
git clone git@github.com:JJXiangJiaoJun/gluon_PyramidBox.git
cd gluon_PyramidBox
```

## Download and prepare data

1. download [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) dataset into `widerface/downloads`

  ```bash
  $$ tree widerface/downloads
  widerface/downloads
  ├── eval_tools.zip
  ├── Submission_example.zip
  ├── wider_face_split.zip
  ├── WIDER_test.zip
  ├── WIDER_train.zip
  └── WIDER_val.zip
  ```

2. Prepare data: unzip data, annotations and eval_tools to `./widerface`
  ```bash
python tool/prepare.py
$$ tree widerface -L 1
widerface
├── downloads
├── eval_tools
├── wider_face_split
├── WIDER_train
└── WIDER_val
  ```

3. Prepare custom val dataset for quick validation (crop and resize to 640)

  ```bash
$$ python tool/build_custom_val.py
$$ tree widerface -L 1
widerface
├── downloads
├── eval_tools
├── WIDER_custom
├── wider_face_split
├── WIDER_train
└── WIDER_val
  ```

## Train on WIDER FACE Datasets
train vgg16 based pyramidbox with 1 gpus.I only implement VGG16 as backbone currently:
```bash
python train_end2end.py --use-bn
```
or you can see more details:
```bash
python train_end2end.py --help
```
## Evalution
eval your own model on WIDER FACE Datasets:
```bash
python eval.py --use-bn --model models/pyramidbox/pyramidbox_best.params
```
## Reference
* [PyramidBox with pytorch](https://github.com/Goingqs/PyramidBox)
* [sfd.gluoncv](https://github.com/yangfly/sfd.gluoncv)