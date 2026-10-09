# 基于 CNN 的 MNIST 手写数字分类器

一个使用 PyTorch 构建的简单卷积神经网络（CNN），用于识别 MNIST 数据集中的手写数字。

## 功能特性

- 首次运行时自动联网下载 MNIST 数据集（通过 `torchvision`）
- CNN 网络结构：2 个卷积层 + 最大池化 + Dropout + 2 个全连接层
- 支持 CUDA 时自动使用 GPU 加速训练
- 完整的训练与评估循环，逐 epoch 输出准确率
- 提供推理脚本，可识别自定义手写数字图片

## 环境安装

```bash
pip install -r requirements.txt
```

## 训练模型

```bash
python train.py --epochs 5
```

训练完成后模型保存为 `mnist_cnn.pt`，预期测试准确率约 99%。

## 预测数字

```bash
python predict.py path/to/digit.png --model-path mnist_cnn.pt
```

## 参数说明

| 参数           | 默认值         | 说明                     |
| -------------- | -------------- | ------------------------ |
| `--epochs`     | 5              | 训练轮数                 |
| `--batch-size` | 64             | 训练批次大小             |
| `--lr`         | 1e-3           | 学习率（Adam 优化器）    |
| `--data-dir`   | ./data         | MNIST 数据集下载/缓存目录 |
| `--model-path` | mnist_cnn.pt   | 模型保存/加载路径        |
