from torchvision import transforms
import numpy as np
from PIL import Image
from skimage.filters import gaussian

def gaussian_blur(image, sigma=20):
    return Image.fromarray(gaussian(image, sigma=sigma, multichannel=True))

def random_flip(image, p=0.5):
    if np.random.rand() < p:
        return transforms.functional.hflip(image)
    return image

def random_rotation(image, degrees=20):
    angle = np.random.uniform(-degrees, degrees)
    return transforms.functional.rotate(image, angle)

def apply_augmentation(image, augmentation_functions):
    augmented_images = [image]
    for aug_function in augmentation_functions:
        augmented_image = aug_function(image)
        augmented_images.append(augmented_image)
    return augmented_images