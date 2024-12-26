# Optimization on Lie groups with proved acceleration

This repository contains the code for the paper [Kong & Tao. Quantitative Convergences of Lie Group Momentum Optimizers. NeurIPS 2023](https://arxiv.org/pdf/2405.20390)

The algorithm works for general compact Lie groups. The code implements algorithms that optimize functions of orthogonal matrices ($SO(n)$, the set of square matrices with orthonormal columns). This family of algorithms is derived from a variational approach and is both accurate and efficient, with proved accelration. Please also see our work for [optimization on Stiefel manifold (non-square matrices with orthonormal columns)](https://arxiv.org/pdf/2205.14173)
### Example:  eigenvalue decomposition
Please run [MomentumLieGD_SOn.ipynb](MomentumLieGD_SOn.ipynb).

### Example: Vision Transformer
For more details, please see also the paper as well as Sec 3.2 in [this paper](https://arxiv.org/pdf/2205.14173).

Please check the folder ViT. Run [ViT_main.py](ViT_main.py). For example: 
```
python ViT_main.py --optim-method LieGRoupSGD_NAG_SC --constraint Across --dataset c10
```
for Lie-NAG-SC optimizer on CIFAR 10.

```
python ViT_main.py --optim-method SGD --dataset c10
```
for SGD optimizer on CIFAR 10 (without orthogonal constraint).

(Modified form the following repositary: [Training process](https://github.com/omihub777/ViT-CIFAR); [model implementation](https://github.com/lucidrains/vit-pytorch))

## Citation
Feel free to cite if you want to use these optimizers in your research!

@article{kong2024quantitative,
  title={{Quantitative Convergences of Lie Group Momentum Optimizers}},
  author={Kong, Lingkai and Tao, Molei},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
