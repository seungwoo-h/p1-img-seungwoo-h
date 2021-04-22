import functools
import operator
import torch
import torch.nn as nn
import torchvision.models as models

class Network(nn.Module):
    def __init__(self, pretrained_model,  out_features, input_dim=(3, 224, 224)):
        super(Network, self).__init__()
        # Load pretrained model, only feature extractors
        self.backbone = nn.Sequential(*(list(pretrained_model.children())[:-1]))
        # Auto-calculate input for the fc layers
        num_features_before_fcnn = functools.reduce(operator.mul, 
                                                    list(self.backbone(torch.rand(1, *input_dim)).shape))
        # Fc layer
        # in features may be 1024? 
        self.fc1 = nn.Sequential(nn.Linear(in_features=num_features_before_fcnn, out_features=out_features),)
        
    def forward(self, x):
        output = self.backbone(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output