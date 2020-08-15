# Introduction

This is a pytorch repository of method ‘A more robust domain feature decoupling network ’(R-DFDN), which is heavily followed by https://github.com/salesforce/corr_based_prediction。

Main contributer：Zechen. Zhao 、Shijie.Li 、Tian.Tian

Our model is more robust and stable compared with Corr-Prediction：

![Model](https://github.com/Complicateddd/R-DFDN/blob/master/temp/model.jpg)

## Usage && Implement

### Requirement：

Python3.6

Pytorch1.2

Numpy1.8

Tqdm

### Run：

#### 1、Generate Colored MNIST:

```python
python gen_color_mnist.py
```

#### 2、Run Correlation based regularization:

```python
python main.py --dataset fgbg_cmnist_cpr0.5-0.5 --seed 0 --root_dir cmnist --save_dir corr --beta 0.1
```

#### 3、Run existing regularization methods:

Maximum Likelihood Estimate (MLE):

```python
python existing_methods.py --dataset fgbg_cmnist_cpr0.5-0.5 --seed 0 --root_dir cmnist --lr 0.0001 --bs 128 --save_dir mle
```

Adaptive Batch Normalization (AdaBN):

```python
python existing_methods.py --dataset fgbg_cmnist_cpr0.5-0.5 --seed 0 --root_dir cmnist --lr 0.0001 --bs 32 --save_dir adabn --bn --bn_eval
```

Clean Logit Pairing (CLP):

```python
python existing_methods.py --dataset fgbg_cmnist_cpr0.5-0.5 --seed 0 --root_dir cmnist --lr 0.0001 --save_dir clp --clp --beta 0.5
```

Projected Gradient Descent (PGD) based adversarial training:

```python
python existing_methods.py --dataset fgbg_cmnist_cpr0.5-0.5 --seed 0 --root_dir cmnist --lr 0.0001 --save_dir pgd --pgd --nsteps 20 --stepsz 2 --epsilon 8
```

Variational Information Bottleneck (VIB):

```python
python existing_methods.py --dataset fgbg_cmnist_cpr0.5-0.5 --seed 0 --root_dir cmnist --lr 0.001 --save_dir inp --inp_noise 0.2
```

#### 4、Run our R-DFDN：

Commands to run R-DFDN:

```python
python train_RDFDN.py --use tf_board True --epochs 300 
```

See the result in tensorboardX:

![tensorboard](https://github.com/Complicateddd/R-DFDN/blob/master/temp/tensorboard.png)

#### 5、Run ADDA \ RevGrad \ WDGRL:

The same as a implement in https://github.com/jvanvugt/pytorch-domain-adaptation

### Some result:

Evaluation on best model forom c-mnist

|                 | CMNIST(B) | MNIST  | MNIST-M | SVHN   | Absolute Gain |
| --------------- | --------- | ------ | ------- | ------ | ------------- |
| Corr-Prediction | 96.88     | 85.94  | 79.69   | 43.75  | +0.00         |
| MLE             | 11.72     | 7.81   | 8.59    | 11.72  | \             |
| ADABN           | 12.5      | 16.41  | 12.5    | 9.38   | \             |
| CLP             | 22.66     | 7.81   | 9.38    | 13.28  | \             |
| PGD             | 18.75     | 13.28  | 8.59    | 7.03   | \             |
| VIB             | 16.41     | 13.28  | 11.72   | 11.72  | \             |
| R-DFDN(ours)    | 96.09     | **97.65**  | **85.16**   | **60.94**  | \ |
|                 | -0.78     | +11.71 | +5.47   | +17.19 | +33.58        |

