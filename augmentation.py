from torchvision import transforms
import numpy as np
from PIL import Image
from skimage.filters import gaussian

def gaussian_blur(image, kernel_size=(19, 19)):
    image_tensor = transforms.ToTensor()(image)
    transform = transforms.GaussianBlur(kernel_size)
    blurred_image_tensor = transform(image_tensor)
    blurred_image = transforms.ToPILImage()(blurred_image_tensor)
    
    return blurred_image

def flip(image):
    return transforms.functional.hflip(image)

def plus_rotate(image, degrees=20):
    return transforms.functional.rotate(image, degrees)

def minus_rotate(image, degrees=-20):
    return transforms.functional.rotate(image, degrees)

def apply_augmentation(image, augmentation_functions):
    augmented_images = [image]
    for aug_function in augmentation_functions:
        augmented_image = aug_function(image)
        augmented_images.append(augmented_image)
    return augmented_images