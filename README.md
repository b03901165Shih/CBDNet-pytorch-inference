# CBDNet-pytorch-inference
This repository contains a Pytorch inference model of CBDNet from:

```
Guo, S., Yan, Z., Zhang, K., Zuo, W., & Zhang, L.
"Toward convolutional blind denoising of real photographs."
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2019).
```

The pre-trained models are produced by converting their original models in [CBDNet](https://github.com/GuoShi28/CBDNet) using [pytorch-mcn](https://github.com/albanie/pytorch-mcn). 


# Usage

Testing code for small images or image patches (a rewrite of "Test_Patches.m" in the original code):
```
python3 Test_Patches.py
```

# Some Results
<p align="left">
  <img width="45%" height="45%" src="./imgs/DND_01.png" />
  <img width="45%" height="45%" src="./imgs/DND_02.png" />
</p>
<p align="left">
  <img width="45%" height="45%" src="./imgs/DND_03.png" />
  <img width="45%" height="45%" src="./imgs/DND_04.png" />
</p>


