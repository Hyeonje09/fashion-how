# augmentation.py

from torchvision import transforms
import numpy as np
from PIL import Image

def horizontal_flip(image):
    return transforms.functional.hflip(image)

def vertical_flip(image):
    return transforms.functional.vflip(image)

def random_rotation(image, degrees=20):
    angle = np.random.uniform(-degrees, degrees)
    return transforms.functional.rotate(image, angle)

def random_color_jitter(image, brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.1), saturation_range=(0.9, 1.1)):
    brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    contrast = np.random.uniform(contrast_range[0], contrast_range[1])
    saturation = np.random.uniform(saturation_range[0], saturation_range[1])
    
    return transforms.functional.adjust_brightness(
        transforms.functional.adjust_contrast(
            transforms.functional.adjust_saturation(image, saturation),
            contrast
        ),
        brightness
    )

def apply_augmentation(image, augmentation_functions):
    augmented_images = []
    if image.shape[0] == 4:  # If image has 4 channels (e.g., RGBA), convert to RGB
        image = image[:3, :, :]
    for aug_function in augmentation_functions:
        augmented_image = aug_function(image)
        augmented_images.append(augmented_image)
    return augmented_images
