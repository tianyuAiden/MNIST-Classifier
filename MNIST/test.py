import torch
from torchvision import transforms
from PIL import Image
from MNIST.MNIST_CNN import MNIST_CNN

# 定义模型结构，和训练时保持一致
model = MNIST_CNN()

# 加载保存的模型权重
model.load_state_dict(torch.load("mnist_epoch_10.pth", map_location=torch.device('cpu')))
model.eval()  # 设置为评估模式

# 图片路径
img_path_1 = "./test_pics/6.png"
img_path_2 = "./test_pics/8.png"

# 图像预处理：转灰度（如果不是），调整大小，转Tensor，归一化
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 保证是1通道
    transforms.Resize((28, 28)),  # 调整大小
    transforms.ToTensor(),  # 转为Tensor，自动归一化到[0,1]
    # MNIST数据集默认不做额外归一化，一般ToTensor后就可以
])

# 载入并预处理图片
image = Image.open(img_path_2)
input_tensor = transform(image)

# 增加 batch 维度，变成 [1, 1, 28, 28]
input_tensor = input_tensor.unsqueeze(0)

# 前向推理
with torch.no_grad():
    outputs = model(input_tensor)
    print("Logits:", outputs.numpy())    # 每个类别的分数
    probs = torch.softmax(outputs, dim=1)
    print("Probs:", probs.numpy())      # 每个类别的概率
    _, predicted = torch.max(outputs, 1)
    print("Predicted:", predicted.item())


print(f"预测数字是: {predicted.item()}")
