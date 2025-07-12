import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
import torchvision.models.segmentation
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
from CACRWNet import *
# from ResNet50_Deeplabv3 import resnet50_DV

from PIL import Image
# PIL_image = Image.fromarray(ndarray_image)   #这里ndarray_image为原来的numpy数组类型的输入

def main():

    model = resnet50_DV()

    pretrain_weights_path = "/tmp/pycharm_project_36/resnet18_new3_all.pth"
    model.load_state_dict(torch.load(pretrain_weights_path))
    target_layers = [model.backbone.group_att]




    data_transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ]
    )

    # data_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(30),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # load image
    img_path = "/tmp/pycharm_project_36/AID/Forest/forest_122.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = None

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    # 调整热图尺寸以匹配图像尺寸
    grayscale_cam_resized = np.array(Image.fromarray(grayscale_cam).resize((img.shape[1], img.shape[0])))

    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam_resized, use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()

