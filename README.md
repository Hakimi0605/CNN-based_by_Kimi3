# CNN-based MNIST Classifier

A simple Convolutional Neural Network built with PyTorch to classify handwritten digits from the MNIST dataset.

## Features

- Automatically downloads the MNIST dataset on first run (via `torchvision`)
- CNN architecture: 2 conv layers + max pooling + dropout + 2 fully connected layers
- GPU acceleration when CUDA is available
- Training and evaluation loops with per-epoch accuracy reporting
- Inference script to predict digits from custom images

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --epochs 5
```

The model is saved to `mnist_cnn.pt`. Expected test accuracy: ~99%.

## Predict

```bash
python predict.py path/to/digit.png --model-path mnist_cnn.pt
```

## Options

| Argument       | Default        | Description                    |
| -------------- | -------------- | ------------------------------ |
| `--epochs`     | 5              | Number of training epochs      |
| `--batch-size` | 64             | Training batch size            |
| `--lr`         | 1e-3           | Learning rate (Adam)           |
| `--data-dir`   | ./data         | MNIST download/cache directory |
| `--model-path` | mnist_cnn.pt   | Where to save/load the model   |
