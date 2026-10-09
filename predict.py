"""Predict the digit in a 28x28 image using the trained MNIST CNN."""

import argparse

import torch
from PIL import Image
from torchvision import transforms

from train import CNN


def main():
    parser = argparse.ArgumentParser(description="Predict a digit from an image")
    parser.add_argument("image", type=str, help="Path to an image of a handwritten digit")
    parser.add_argument("--model-path", type=str, default="mnist_cnn.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    image = transform(Image.open(args.image)).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(image), dim=1)
    pred = probs.argmax(1).item()
    print(f"Predicted digit: {pred} (confidence {probs[0, pred]:.2%})")


if __name__ == "__main__":
    main()
