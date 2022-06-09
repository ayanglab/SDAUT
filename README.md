# SDAUT

by Jiahao Huang (j.huang21@imperial.ac.uk)

This is the official implementation of our proposed SDAUT:

Swin Deformable Attention U-Net Transformer (SDAUT) for Explainable Fast MRI

Please cite:

```
xxx
```

The structure of SDAUT:

![Overview_of_SDAUT](./tmp/files/SDAUT_1.png)


Our proposed swin deformable self-attention:

![Overview_of_SDAUT](./tmp/files/SDAUT_2.png)


### Calgary Campinas multi-channel dataset (CC) 

To train SDAUT on CC:

`python main_train_sdaut.py --opt ./options/SDAUT/train_sdaut_CC_G1D30.json`

To test SDAUT on CC:

`python main_test_sdaut.py --opt ./options/SDAUT/test/test_sdaut_CC_G1D30.json`

This repository is based on:

Swin Transformer for Fast MRI 
([code](https://github.com/ayanglab/SwinMR) and [paper](https://arxiv.org/abs/2201.03230));

SwinIR: Image Restoration Using Swin Transformer 
([code](https://github.com/JingyunLiang/SwinIR) and [paper](https://arxiv.org/abs/2108.10257));

Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
([code](https://github.com/microsoft/Swin-Transformer) and [paper](https://arxiv.org/abs/2103.14030)).

Vision Transformer with Deformable Attention
([code](https://github.com/LeapLabTHU/DAT) and [paper](https://arxiv.org/abs/2201.00520)).