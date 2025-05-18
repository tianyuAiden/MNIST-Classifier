from torchvision import datasets, transforms

# download dataset
train_set = datasets.MNIST("dataset", train=True, download=True, transform=transforms.ToTensor())
test_set = datasets.MNIST("dataset", train=False, download=True, transform=transforms.ToTensor())

print("train set size: ", len(train_set))
print("test set size: ", len(test_set))

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
