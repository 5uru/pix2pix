from PIL import Image
import os
import numpy as np
from tinygrad import Tensor

# Constants for image processing
IMG_SIZE = 256
IMG_CHANNELS = 3
NORM_FACTOR = 127.5

def load_image(path):
    """
    Load an image and split it into left and right halves.
    """
    img = Image.open(path).convert("RGB")
    # Split the image into two: 512x256 to 256x256
    width, height = img.size
    left = img.crop((0, 0, width // 2, height))
    right = img.crop((width // 2, 0, width, height))

    return np.array(left, dtype=np.float32), np.array(right, dtype=np.float32)

def normalize(input_image, real_image):
    """
    Normalize images to range [-1, 1] by dividing by 127.5 and subtracting 1.
    """
    input_image = (input_image / NORM_FACTOR) - 1
    real_image = (real_image / NORM_FACTOR) - 1

    return input_image, real_image

def load_all_images_from_folder(folder):
    """
    Load all images from a folder, split them into pairs and normalize.
    """
    input_images = []
    real_images = []

    # Check if folder exists
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    # Process all image files in the folder
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(folder, filename)
            input_image, real_image = load_image(file_path)
            input_image, real_image = normalize(input_image, real_image)
            input_images.append(input_image)
            real_images.append(real_image)

    print(f"Loaded {len(input_images)} images from {folder}")
    return np.array(input_images), np.array(real_images)

def get_dataset(batch_size=32):
    """
    Load and prepare training and validation datasets.
    """
    # Load training dataset
    print("Loading training dataset...")
    train_input_images, train_real_images = load_all_images_from_folder("cityscapes/train")

    # Calculate how many complete batches we can make
    batch_count = len(train_input_images) // batch_size
    if batch_count == 0:
        raise ValueError(f"Batch size {batch_size} is larger than training set size {len(train_input_images)}")

    # Convert to Tensors and reshape to NCHW format (batch, channels, height, width)
    train_input_images = Tensor(train_input_images).reshape(-1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    train_real_images = Tensor(train_real_images).reshape(-1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

    # Organize into batches
    batch_input_images = train_input_images[:batch_count * batch_size].reshape(batch_count, batch_size, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    batch_real_images = train_real_images[:batch_count * batch_size].reshape(batch_count, batch_size, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    train_dataset = (batch_input_images, batch_real_images)

    # Load validation dataset
    print("Loading validation dataset...")
    test_input_images, test_real_images = load_all_images_from_folder("cityscapes/val")
    test_input_images = Tensor(test_input_images).reshape(-1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    test_real_images = Tensor(test_real_images).reshape(-1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

    # Organize validation data into batches
    batch_count = len(test_input_images) // batch_size
    test_input_images = test_input_images[:batch_count * batch_size].reshape(batch_count, batch_size, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    test_real_images = test_real_images[:batch_count * batch_size].reshape(batch_count, batch_size, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    test_dataset = (test_input_images, test_real_images)

    return train_dataset, test_dataset

if __name__ == "__main__":
    train_dataset, test_dataset = get_dataset()
    print("Train dataset shapes:", train_dataset[0].shape, train_dataset[1].shape)
    print("Test dataset shapes:", test_dataset[0].shape, test_dataset[1].shape)