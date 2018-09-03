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
- install matplotlib
```
pip install matplotlib
```
- install spicy
```
pip install spicy
```

 ## CycleGAN train/test
- Download Datasets
```
bash ./datasets/download_cyclegan_dataset.sh maps
```
- Train Model
```
CUDA_VISIBLE_DEVICES=0 model.py
```


 The test results will be saved to html file: ```./results/maps_cyclegan/latest_test/index.html```
 ## Acknowledgments
This code and Github ideas are inspired by [pytorch-CycleGan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#cyclegan-and-pix2pix-in-pytorch), [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN/),[Tensorflow-GAN](https://github.com/xhujoy/CycleGAN-tensorflow/)

