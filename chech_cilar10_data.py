import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib

# SSH / 服务器环境下使用非交互式后端
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


classes = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)


print("训练集样本数:", len(train_dataset))
print("测试集样本数:", len(test_dataset))

images, labels = next(iter(train_loader))

print("一个 batch 的图像 shape:", images.shape)
print("一个 batch 的标签 shape:", labels.shape)
print("前 10 个标签编号:", labels[:10].tolist())
print("前 10 个标签名称:", [classes[i] for i in labels[:10]])


def imshow(img):
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
    std = torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)

    img = img * std + mean
    img = torch.clamp(img, 0, 1)

    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.axis("off")


# 创建结果目录
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10, 4))

for i in range(8):
    plt.subplot(2, 4, i + 1)
    imshow(images[i])
    plt.title(classes[labels[i]])
    plt.axis("off")

plt.tight_layout()

save_path = "results/cifar10_samples.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"样本图片已保存到: {save_path}")
