This implementation in Keras is based on [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#cyclegan-and-pix2pix-in-pytorch)
 ![EXAMPLE](https://camo.githubusercontent.com/69cbc0371777fba5d251a564e2f8a8f38d1bf43f/68747470733a2f2f6a756e79616e7a2e6769746875622e696f2f4379636c6547414e2f696d616765732f7465617365725f686967685f7265732e6a7067)
 ## Prerequisites
-Windows or Linux or macOS
-Python 3
-GPU + CUDA CuDNN

## Installation
- install Keras
``` pip install keras ```
 ```
pip install keras
```
 - install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)
``` pip install visdom dominate ```
 ```
pip install visdom dominate
```
 ## CycleGAN train/test
- Download Datasets
```
bash ./datasets/download_cyclegan_dataset.sh maps
```
-Before training you have to run ```python -m visdom.server``` and to see the training result click the URL http://localhost:8097/
- Train Model
```
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
-Test Model
```
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
 The test results will be saved to html file: ```./results/maps_cyclegan/latest_test/index.html```
 ## Acknowledgments
This code and Github ideas are inspired by [pytorch-CycleGan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#cyclegan-and-pix2pix-in-pytorch)