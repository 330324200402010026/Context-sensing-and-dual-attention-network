import torch
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as nppip
import matplotlib.pyplot as plt
from CACRWNet import *
#from ResNet50_Deeplabv3 import *
#from ResNet50 import *
#from resnet18 import *conda install seaborn

import shutil
import config
#from split import split
import seaborn as sns
from sklearn.metrics import confusion_matrix


#数据增强
# data_transform = {
#     # 数据预处理
#     "train": transforms.Compose(
#         [
#         transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),  # 随机裁剪到156*156
#         transforms.RandomRotation(degrees=45),  # 随机旋转
#         transforms.RandomHorizontalFlip(),  # 随机水平翻转
#         transforms.CenterCrop(size=224),  # 中心裁剪到124*124
#         transforms.ToTensor(),  # 转化成张量
#         transforms.Normalize([0.485, 0.456, 0.406],  # 归一化
#                              [0.229, 0.224, 0.225])
#          ]),
#
#     "val": transforms.Compose(
#         [transforms.Resize(256),
#          transforms.CenterCrop(224),
#          transforms.ToTensor(),
#          transforms.Normalize([0.485, 0.456, 0.406],
#                               [0.229, 0.224, 0.225])
#                                ])}

data_transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#利用Dataloader加载数据
train_directory = config.TRAIN_DATASET_DIR
valid_directory = config.VALID_DATASET_DIR

batch_size = config.BATCH_SIZE
num_classes = config.NUM_CLASSES

#from torchvision.datasets import ImageFolder

train_datasets = datasets.ImageFolder('/tmp/pycharm_project_36/AIDdatasets28/train',
                                      is_valid_file=lambda x: x.endswith(('.png','.jpg','.jpeg','.tif','.tiff')),
                                      transform=data_transform["train"])

valid_datasets = datasets.ImageFolder('/tmp/pycharm_project_36/AIDdatasets28/val',
                                      is_valid_file=lambda x: x.endswith(('.png','.jpg','.jpeg','.tif','.tiff')),
                                      transform=data_transform["val"])

# 加载验证集

train_data_size = len(train_datasets)
train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

valid_data_size = len(valid_datasets)
valid_data = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)

# print(train_data_size, valid_data_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #若有gpu可用则用gpu
#使用Resnet-50的预训练模型进行迁移学习
# model = resnet50_DV()
# model = resnet50()
model = resnet50_DV()

model.to(device)
# testmodel = model
#查看更改后的模型参数
#print('after:{%s}\n'%resnet50)

#定义损失函数和优化器
# loss_func = nn.NLLLoss()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=3e-4, amsgrad=True, weight_decay=0.0001)

#训练过程
def train_and_valid(model, loss_function, optimizer, epochs, e):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #若有gpu可用则用gpu
    # scaler = torch.cuda.amp.GradScaler()
    # autocast = torch.cuda.amp.autocast

    # print("using {} device.".format(device))
    # print("PyTorch Version: ", torch.__version__)
    print("using {} images for training, {} images for validation.".format(
        train_data_size, valid_data_size))
    params = sum([v.numel() for k, v in model.state_dict().items()])
    # 打印信息
    print(params)
    record = []
    best_acc = 0.0
    Bestacc = []
    best_epoch = 0
    writer = SummaryWriter('logs')
    writer_acc = SummaryWriter('acc')

    for epoch in range(epochs): #训练epochs轮
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train() #训练

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        train_bar = tqdm(train_data, file=sys.stdout)
        for i, (inputs, labels) in enumerate(train_bar):

            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(labels)
            optimizer.zero_grad()

            # with autocast():
            optimizer.zero_grad() #梯度清零

            outputs = model(inputs) #数据前馈，正向传播

            loss = loss_function(outputs, labels) #输出误差

            # scaler.scale(loss).backward()
            #
            # scaler.step(optimizer)
            #
            # scaler.update()
            #
            loss.backward() #反向传播

            optimizer.step() #优化器更新参数

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)
            train_bar.desc = "train epoch[{}/{}]——loss:{:.3f}".format(epoch + 1,
                                                                      epochs,
                                                                      loss)

        with torch.no_grad():
            model.eval() #验证
            val_bar = tqdm(valid_data, file=sys.stdout)
            for j, (inputs, labels) in enumerate(val_bar):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)
                val_bar.desc = "valid epoch——[{}/{}]".format(epoch + 1,
                                                             epochs)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        writer.add_scalar('train', avg_train_loss, global_step=epoch)
        writer_acc.add_scalar('acc', avg_valid_acc, global_step=epoch)

        record.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if avg_valid_acc > best_acc  : #记录最高准确性的模型
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            torch.save(model, f'{e+5}_model.pth')
            # torch.save(model.state_dict(), 'resnet50meiyoue_model_onlyweigths.pth')。
            torch.save(model.state_dict(), 'resnet18_new5_all.pth')
        # if  avg_train_acc > 98.4:  # 记录最高准确性的模型
        #     best_acc = avg_train_acc
        #     best_epoch = epoch + 1
        #     torch.save(model, f'{e + 1}_model.pth')


        epoch_end = time.time()

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
    Bestacc = best_acc

        # torch.save(model, 'trained_models/resnet50_model_' + str(epoch + 1) + '.pth')
    return model, record, Bestacc

            # -------------------- 热力图生成 --------------------


# 修改后的热力图生成部分
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.target_layer = model.online_encoder.layer4[2].conv3
        self.activations = []
        self.gradients = []

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output.detach())

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())

    def __call__(self, x):
        self.activations = []
        self.gradients = []

        # 前向传播
        features = self.model.online_encoder(x.unsqueeze(0))
        class_idx = features.argmax(dim=1)

        # 反向传播
        self.model.zero_grad()
        features[0, class_idx].backward(retain_graph=True)

        # 计算热力图
        activations = self.activations[0]  # (1, 2048, 7, 7)
        gradients = self.gradients[0]  # (1, 2048, 7, 7)

        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)  # (1, 2048, 1, 1)
        weights = torch.abs(weights)  # 新增此行
        heatmap = torch.sum(activations * weights, dim=1)  # (1, 7, 7)
        #heatmap = F.relu(heatmap)
        heatmap = F.relu(heatmap.squeeze(0))

        # # 维度修正（关键修改）
        # if heatmap.dim() == 3:
        #     heatmap = heatmap.squeeze(0)  # 移除batch维度 (7, 7)

        # 插值到输入尺寸
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),  # 添加batch和channel维度 -> (1,1,7,7)
            size=224,
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        # 标准化
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        return heatmap
        #heatmap = torch.einsum('bchw,bc->bhw', activations, weights)

#结果
if __name__=='__main__':
    num_epochs = config.NUM_EPOCHS
    num = 1
    # Best = 0
    # list = []
    for epoch in range(num):  # 训练epochs轮
        print("Experiment: {}/{}".format(epoch + 1, num))
        trained_model, record, Best = train_and_valid(model, loss_func, optimizer, num_epochs, epoch)
        # list.append(Best)
        # print('delete dataset')
        # shutil.rmtree('./dataset')
        # print('creat dataset')
        # split()
    # print(list)
    # print('mean', np.mean(list))
        torch.save(record, config.TRAINED_MODEL)

    # record = np.array(record)
    # plt.plot(record[:, 0:2])
    # plt.legend(['Train Loss', 'Valid Loss'])
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Loss')
    # plt.ylim(0, 1)
    # plt.savefig('loss.png')
    # plt.show()
    #
    # plt.plot(record[:, 2:4])
    # plt.legend(['Train Accuracy', 'Valid Accuracy'])
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Accuracy')
    # plt.ylim(0, 1)
    # plt.savefig('accuracy.png')
    # plt.show()
#
# '''
#     model = torch.load('trained_models/resnet50_model_23.pth')
#     predict(model, '61.png')
# '''


    # 生成热力图示例
    grad_cam = GradCAM(model)
    sample_img, _ = valid_datasets[0]
    heatmap = grad_cam(sample_img.to(device))

    plt.imshow(heatmap)
    plt.title("Class Activation Map")
    plt.show()


    # 生成混淆矩阵
    # 使用微调后的模型生成预测
    def get_predictions(model, loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        return all_labels, all_preds


    # y_true, y_pred = get_predictions(finetune_model, test_loader)
    # classes = ["Class1", "Class2", "Class3", ...]  # 替换为实际类别名
    # cm = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()
    y_true, y_pred = get_predictions(model, valid_data)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=num_classes, yticklabels=num_classes,
                cmap='Blues', cbar=False)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.show()

