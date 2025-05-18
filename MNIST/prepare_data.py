"""
MNIST CNN Training Script

1. 准备数据
   - 加载 MNIST 训练集和测试集
   - 使用 DataLoader 将数据打包成小批量

2. 创建模型实例
   - 实例化 MNIST_CNN 类

3. 定义损失函数
   - 使用 CrossEntropyLoss 计算分类误差

4. 定义优化器
   - 如 SGD 或 Adam

5. 训练循环
   for epoch in range(num_epochs):
       for inputs, labels in train_loader:
           # 前向传播：model(inputs) → outputs
           # 计算损失：criterion(outputs, labels)
           # 反向传播：loss.backward()
           # 更新参数：optimizer.step()
           # 清零梯度：optimizer.zero_grad()

6. 验证/测试
   - 在每个 epoch 结束后，用 test_loader 评估模型准确率
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # 用于 F.relu
from tqdm import tqdm  # tqdm进度条

from MNIST.MNIST_CNN import MNIST_CNN

# 选择设备：优先 GPU (CUDA 或 MPS)，否则 CPU
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")
print(f"Using device: {device}")

# download dataset
train_set = datasets.MNIST("dataset", train=True, download=True, transform=transforms.ToTensor())
test_set = datasets.MNIST("dataset", train=False, download=True, transform=transforms.ToTensor())

print("train set size: ", len(train_set))
print("test set size: ", len(test_set))

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

# creat module
mnist = MNIST_CNN().to(device)  # 把模型移动到 device 上

# defined loss function
loss_function = nn.CrossEntropyLoss()

# defined optimizer
optimizer = optim.Adam(mnist.parameters(), lr=0.001)
# lr：Learning rate

#  begin training
# 设置训练的轮数
num_epochs = 10

# 开始训练循环，每个 epoch 对整个训练集进行一次完整遍历
for epoch in range(num_epochs):
    mnist.train()  # 切换到训练模式，会启用 dropout、batch norm 等训练特性
    running_loss = 0.0  # 用于累加本 epoch 的总损失

    # tqdm进度条，包装train_loader，显示进度和损失
    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

    # 遍历训练数据集中的每个 batch
    for inputs, labels in loop:
        # 把数据也移动到 device 上
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # 清零上一轮梯度，避免累加
        outputs = mnist(inputs)  # 前向传播，计算模型预测值
        loss = loss_function(outputs, labels)  # 计算当前 batch 的损失
        loss.backward()  # 反向传播，计算参数梯度
        optimizer.step()  # 更新参数

        # 累加：loss.item() 是当前 batch 的平均损失，乘以 batch 大小得到总损失
        running_loss += loss.item() * inputs.size(0)

        # 更新tqdm显示当前batch的loss
        loop.set_postfix(loss=loss.item())

    torch.save(mnist.state_dict(), f"mnist_epoch_{epoch + 1}.pth")

    # 计算并打印本 epoch 的平均损失
    epoch_loss = running_loss / len(train_set)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # ===== 测试/验证阶段 =====
    mnist.eval()  # 切换到评估模式，关闭 dropout、使用累计的 batch norm 参数
    correct = total = 0  # 用于统计测试集上的正确预测数和总样本数

    # 在评估时不需要计算梯度，加速计算并节省显存
    with torch.no_grad():
        for inputs, labels in test_loader:
            # 同样移动到 device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = mnist(inputs)  # 前向传播
            _, preds = torch.max(outputs, 1)  # 取每行最大值的索引，作为预测类别
            correct += (preds == labels).sum().item()  # 累加正确预测数
            total += labels.size(0)  # 累加总样本数

    # 计算并打印测试集准确率
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}\n")
