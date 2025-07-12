import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import cv2
import numpy as np


# ----------------------
# 1. 定义特征提取工具类
# ----------------------
class FeatureExtractor:
    def __init__(self, model_name='resnet50', target_layers=['layer1', 'layer2', 'layer3', 'layer4']):
        self.model = self._build_model(model_name)
        self.feature_maps = {}
        self._register_hooks(target_layers)

    def _build_model(self, name):
        """加载预训练模型"""
        model_dict = {
            'resnet18': models.resnet18(pretrained=True),
            'resnet34': models.resnet34(pretrained=True),
            'resnet50': models.resnet50(pretrained=True)
        }
        return model_dict[name]

    def _register_hooks(self, target_layers):
        """为指定层注册前向钩子"""

        def get_activation(name):
            def hook(model, input, output):
                self.feature_maps[name] = output.detach()

            return hook

        for name, layer in self.model.named_modules():
            if name in target_layers:
                layer.register_forward_hook(get_activation(name))

    def preprocess_image(self, img_path, img_size=224):
        """图像预处理"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return transform(img).unsqueeze(0)  # 添加batch维度

    def visualize_features(self, layer_name, n_cols=8, save_path=None):
        """可视化指定层的特征图"""
        if layer_name not in self.feature_maps:
            raise ValueError(f"Layer {layer_name} not found in registered layers")

        features = self.feature_maps[layer_name]
        n_channels = features.size(1)

        # 选择前n_cols个通道可视化
        selected = features[0, :n_cols].unsqueeze(1)  # [C,1,H,W]

        grid = make_grid(selected, nrow=n_cols, normalize=True, scale_each=True)
        grid = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(15, 3))
        plt.imshow(grid)
        plt.title(f"Feature Maps @ {layer_name}")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()


# ----------------------

if __name__ == "__main__":
    # 初始化提取器（以ResNet50为例）
    extractor = FeatureExtractor(
        model_name='resnet50',
        target_layers=['layer1', 'layer2', 'layer3', 'layer4']
    )

    # 预处理图像
    input_tensor = extractor.preprocess_image(r'/tmp/pycharm_project_685/8a1d513097c47953bf37d7b75cdec47.jpg')

    # 前向传播获取特征
    with torch.no_grad():
        extractor.model.eval()
        _ = extractor.model(input_tensor)

    # 可视化各层特征图（保存到当前目录）
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    for layer in layers:
        extractor.visualize_features(
            layer_name=layer,
            n_cols=8,
            save_path=f"{layer}_features.png"
        )
