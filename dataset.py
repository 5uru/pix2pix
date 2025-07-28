from PIL import Image
import os
from tinygrad import Tensor
import numpy as np


def load_image(path):
    img = Image.open(path).convert("RGB")
    # Split the image in two: 512x256 to 256x256
    width, height = img.size
    left = img.crop((0, 0, width // 2, height))
    right = img.crop((width // 2, 0, width, height))

    return np.array(left, dtype=np.float32), np.array(right, dtype=np.float32)

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

def load_all_images_from_folder(folder):
    input_images = []
    real_images = []

    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_image, real_image = load_image(os.path.join(folder, filename))
            input_image, real_image = normalize(input_image, real_image)
            input_images.append(input_image)
            real_images.append(real_image)

    return np.array(input_images), np.array(real_images)

def get_dataset():
    # Train dataset
    train_input_images, train_real_images = load_all_images_from_folder("cityscapes/train")
    train_input_images = Tensor(train_input_images).reshape(-1, 3, 256, 256)
    train_real_images = Tensor(train_real_images).reshape(-1, 3, 256, 256)
    train_dataset = (train_input_images, train_real_images)

    # Test dataset
    test_input_images, test_real_images = load_all_images_from_folder("cityscapes/val")
    test_input_images = Tensor(test_input_images).reshape(-1, 3, 256, 256)
    test_real_images = Tensor(test_real_images).reshape(-1, 3, 256, 256)
    test_dataset = (test_input_images, test_real_images)
    return train_dataset, test_dataset

if __name__ == "__main__":
    train_dataset, test_dataset = get_dataset()
    print("Train dataset shapes:", train_dataset[0].shape, train_dataset[1].shape)
    print("Test dataset shapes:", test_dataset[0].shape, test_dataset[1].shape)
