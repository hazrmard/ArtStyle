import torch
from torch.utils import model_zoo
from torchvision.models import vgg
from torchvision.models.vgg import model_urls
from torchvision.models.vgg import VGG as OldVGG


class VGGFactory:
    """
    Factory class to create `torchvision.models.vgg` modules with support for
    pre-trained weights with custom number of `num_classes`.
    """

    def __init__(self, constructor):
        self.constructor = constructor


    def __call__(self, pretrained=False, model_dir=None, **kwargs):
        model = self.constructor(pretrained=False, **kwargs)
        if pretrained:
            name = self.constructor.__name__.split('.')[-1]
            state = model_zoo.load_url(model_urls[name],
                                        model_dir=model_dir, map_location='cpu')
            wkey = 'classifier.' + str(len(list(model.classifier.children()))-1) + '.weight'
            bkey = wkey[:-6] + 'bias'
            state[wkey] = model.state_dict()[wkey]
            state[bkey] = model.state_dict()[bkey]
            model.load_state_dict(state)
        return model



vgg11 = VGGFactory(vgg.vgg11)
vgg11_bn = VGGFactory(vgg.vgg11_bn)
vgg13 = VGGFactory(vgg.vgg13)
vgg13_bn = VGGFactory(vgg.vgg13_bn)
vgg16 = VGGFactory(vgg.vgg16)
vgg16_bn = VGGFactory(vgg.vgg16_bn)
vgg19 = VGGFactory(vgg.vgg19)
vgg19_bn = VGGFactory(vgg.vgg19_bn)
