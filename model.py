import torch
import torchvision
from torch import nn
from efficientnet_pytorch import EfficientNet


def effb2_model(num_classes:int=3, seed:int=42):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    effb2_transform = weights.transforms()
    effb2_model = EfficientNet.from_pretrained('efficientnet-b2')
    # Freeze all layers in base model
    for param in effb2_model.parameters():
        param.requires_grad = False

    # Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    effb2_model._fc = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=effb2_model._fc.in_features, out_features=num_classes, bias=True),
    )
    
    return effb2_model, effb2_transform