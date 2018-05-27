# Stochastic Cubic Adjusted Gradient Descent (SCAG)

This repository contains pytorch code that produces the  Stochastic Cubic Adjusted Gradient Descent (SCAG) in the paper: [Making Use of Second-order Information: Cubic Regularization for Training DNNs](https://arxiv.org/abs/1712.03950). 

This repository contains example code for ([cifar10](https://www.cs.toronto.edu/~kriz/cifar.html), [cifar100](https://www.cs.toronto.edu/~kriz/cifar.html)) using ([alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), [vgg](https://arxiv.org/abs/1409.1556), [resnet](https://arxiv.org/abs/1512.03385), [wide_resnet](https://arxiv.org/abs/1605.07146)).

## Usage examples
`python main.py --arch model_name --dataset dataset_name`, where `model_name` could be `alexnet`, `vgg`, `wide_resnet`, and `resnet`, `dataset_name` could be `cifar10` and `cifar100`.

For example
```bash
python main.py --arch vgg --dataset cifar10;
python main.py --arch wide_resnet --dataset cifar10;
python main.py --arch alexnet --dataset cifar100;
python main.py --arch wide_resnet --dataset cifar100;
```
## Learning rate adjustment
The learning rate is set as `0.5` at the beginning and decay `0.1` every `50` epochs, other adjustment schemes can be applied for better performance.

Initial learning rate `lr` can be set as follows,
```bash
python main.py --arch vgg --dataset cifar10 --lr 0.5;
```
