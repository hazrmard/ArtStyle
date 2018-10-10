import torch
from torch.utils import model_zoo
from torchvision.models.alexnet import model_urls
from torchvision.models.alexnet import AlexNet as OldAlexNet



class AlexNet(OldAlexNet):
    """
    Modified version of AlexNet with 136 classes to work with the WikiArt dataset.
    If `pretrained=True` loads all AlexNet weights except the very last linear
    layer which has 136 units.

    During training, does not modify convolutional layer weights.

    Has two submodules: `features` and `classifier`.

    Args:

    * model_dir (str): Save path for pre-trained weights for AlexNet.
    * pretrained (bool): Whether to download and use pre-trained weights
    """

    def __init__(self, model_dir: str = None, pretrained: bool = False):
        super().__init__(num_classes=136)
        # load weights for old
        if pretrained:
            state = model_zoo.load_url(model_urls['alexnet'],
                                        model_dir=model_dir, map_location='cpu')
            # replace OldAlexNet final layer parameters with those for 136 classes
            state['classifier.6.weight'] = self.state_dict()['classifier.6.weight']
            state['classifier.6.bias'] = self.state_dict()['classifier.6.bias']
            self.load_state_dict(state)


    def forward(self, x):
        # do not backpropagate gradients to feature detectors
        with torch.no_grad():
            x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

