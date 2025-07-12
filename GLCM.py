
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops
import math as mt

# 读取图像
image = io.imread('/tmp/pycharm_project_685/7b058bdb9c3c3c1a4c821f07d45581b.jpg')

# 转换为灰度图像
if len(image.shape) == 3:  # 如果是彩色图像
    gray_image = color.rgb2gray(image) * 255  # 转换为灰度图像并归一化到 0 - 255
else:  # 如果已经是灰度图像
    gray_image = image.copy()

# 确保灰度值范围在 [0, 255]
gray_image = gray_image.astype(np.uint8)

# 显示灰度图像
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# 设置 GLCM 参数
distances = [1, 2]  # 尝试较小的距离
angles = [0, mt.pi / 4, mt.pi / 2, 3 * mt.pi / 4]  # 常见的角度
levels = 256  # 灰度级数

# 计算 GLCM
glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

# 提取纹理特征
contrast = graycoprops(glcm, 'contrast')
dissimilarity = graycoprops(glcm, 'dissimilarity')
homogeneity = graycoprops(glcm, 'homogeneity')
energy = graycoprops(glcm, 'energy')
correlation = graycoprops(glcm, 'correlation')

# 创建示意图
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

# 显示 GLCM，设置颜色范围
axes[1].imshow(glcm[:, :, 0, 0], cmap='gray', vmin=0, vmax=1)
axes[1].set_title('GLCM (distance=1, angle=0)')
axes[1].axis('off')

# 显示纹理特征值
feature_text = f'Contrast: {contrast[0][0]:.2f}\nDissimilarity: {dissimilarity[0][0]:.2f}\nHomogeneity: {homogeneity[0][0]:.2f}\nEnergy: {energy[0][0]:.2f}\nCorrelation: {correlation[0][0]:.2f}'
plt.figtext(0.5, 0.01, feature_text, ha='center', fontsize=10)

plt.tight_layout()
plt.show()
