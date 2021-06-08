#from .hourglass import HourglassNet
#from .hrnet import HRNet
#from .regnet import RegNet
#from .res2net import Res2Net
from .resnet import ResNet, ResNetV1d, Bottleneck
#from .resnext import ResNeXt
#from .ssd_vgg import SSDVGG

__all__ = ['ResNet', 'ResNetV1d', 'Bottleneck']

#__all__ = [
#    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
#    'HourglassNet', 'Bottleneck'
#]
