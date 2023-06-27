# Bicubic++: Slim, Slimmer, Slimmest - Designing an Industry-Grade Super-Resolution Network (🚀 Winner of [NTIRE RTSR Challange Track 2 (x3 SR)](https://codalab.lisn.upsaclay.fr/competitions/10228) @ CVPR 2023)

##**note**:**This is an unofficial code repo**,some code is inherited from [bicubic-plusplus](https://github.com/aselsan-research-imaging-team/bicubic-plusplus),this method adds ~1dB on Bicubic upscaling PSNR for all tested SR datasets and runs with ~0.5ms(2000FPS) per image on RTX4090,only inference time is included, not pre-processing and post-processing time.

## Installation

`git clone https://github.com/Ysnower/bicubic-plusplus.git`

2.Install the dependencies.
Python 3.7
PyTorch 1.13.1
opencv-python 4.5.1.48
onnxruntime 1.10.0
onnxruntime-gpu 1.10.0

my gpu is RTX4090 with CUDA 12.1

## Data Preparation

only train on DIV2K dataset,you can use your own data, the number of hr images must be equal to lr images.
Download [DIV2K data](https://pan.baidu.com/s/1OxiN6f2FG98A1Rt46UWXGg) `password:xb5r` and unzip the `DIV2K.zip` to the `datasets`. I used 860 images for training and 40 images for validation.

## Train

1.Set validation & training dataset paths in `configs/conf.yaml` (`data.val.lr_path`, `data.val.hr_path`, `data.train.lr_path`, `data.train.hr_path`). Set `loader.train.batch_size` and `loader.val.batch_size` according to your dataset.

**note**: if use degradation(blur is True or img_noise is True), training will slow down,you can use degradation offline to generate lr data to the lr_path.

The scale/sr_rate must be divisible by patch_cropsize.

2.Run train code.

`python3 train.py`

## pytorch inference

`python3 inference.py `

You can change the `img_path` in the inference.py file to your image location.

## onnx inference

1.pytorch to onnx model

`python3 torch2onnx.py `

You can change the `model_path` to your pytorch model location and change the `export_onnx_file` to your onnx model name.

2.onnxruntime inference

`python3 onnx_inference.py`

## reference

1.[bicubic-plusplus](https://github.com/aselsan-research-imaging-team/bicubic-plusplus)
2.[pytorch-static-quant](https://github.com/Ysnower/pytorch-static-quant)

##### **If this repository helps you，please star it. Thanks.**

