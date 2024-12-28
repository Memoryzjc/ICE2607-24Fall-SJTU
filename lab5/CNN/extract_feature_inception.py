import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader

print('Load model: InceptionV3')
model = torch.hub.load('pytorch/vision', 'inception_v3', pretrained=True)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

trans = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    normalize,
])

def features_inception(x):
    x = model.Conv2d_1a_3x3(x)
    x = model.Conv2d_2a_3x3(x)
    x = model.Conv2d_2b_3x3(x)
    x = model.Conv2d_3b_1x1(x)
    x = model.Conv2d_4a_3x3(x)
    x = model.Mixed_5b(x)
    x = model.Mixed_5c(x)
    x = model.Mixed_5d(x)
    x = model.Mixed_6a(x)
    x = model.Mixed_6b(x)
    x = model.Mixed_6c(x)
    x = model.Mixed_6d(x)
    x = model.Mixed_6e(x)
    x = model.Mixed_7a(x)
    x = model.Mixed_7b(x)
    x = model.Mixed_7c(x)
    return x


def extract_features_inception(input_image_path, save_path):
    print('Prepare image data!')
    test_image = default_loader(input_image_path)
    input_image = trans(test_image)
    input_image = torch.unsqueeze(input_image, 0)

    print('Extract features!')
    start = time.time()
    image_feature = features_inception(input_image)
    image_feature = image_feature.detach().numpy()
    print('Time for extracting features: {:.2f}'.format(time.time() - start))

    print('Save features!')
    np.save(save_path, image_feature)