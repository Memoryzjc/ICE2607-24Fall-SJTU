# SJTU EE208

import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader

print('Load model: ResNet50')
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

def features_resnet50(x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)

    return x


def extract_features_resnet50(input_image_path, save_path):
    print('Prepare image data!')
    test_image = default_loader(input_image_path)
    input_image = trans(test_image)
    input_image = torch.unsqueeze(input_image, 0)
    print('Extract features!')

    start = time.time()
    image_feature = features_resnet50(input_image)
    image_feature = image_feature.detach().numpy()
    print('Time for extracting features: {:.2f}'.format(time.time() - start))

    print('Save features!')
    np.save(save_path, image_feature)