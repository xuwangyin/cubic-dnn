import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.srelu import SRelu


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            SRelu(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            SRelu(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            SRelu(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            SRelu(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            SRelu(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

class AlexNetC1(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNetC1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 384, bias=False),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(384, 192, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.classifier(x)
        return x