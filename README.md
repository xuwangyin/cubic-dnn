This repository contains pytorch code that produces the  Stochastic Cubic Adjusted Gradient Descent (SCAG) in the paper: [Making Use of Second-order Information: Cubic Regularization for Training DNNs](https://arxiv.org/abs/1712.03950).

## Usage examples
`python main.py --arch model_name --dataset dataset_name`, where `model_name` could be `alexnet`, `vgg`, `wide_resnet`, and `resnet`, `dataset_name` could be `cifar10` and `cifar100`.

For example
```bash
python main.py --arch vgg --dataset cifar10;
python main.py --arch wide_resnet --dataset cifar10;
python main.py --arch alexnet --dataset cifar100;
python main.py --arch wide_resnet --dataset cifar100;
```
