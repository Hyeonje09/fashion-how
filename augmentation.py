from torchvision import transforms
from PIL import ImageFilter, ImageEnhance
import random

def gaussian_blur(image, kernel_size=3):
    image_pil = image.filter(ImageFilter.GaussianBlur(kernel_size))
    return image_pil

def vertical_flip(image):
    return transforms.functional.vflip(image)

def random_rotate(image, degrees=20):
    return transforms.RandomRotation(degrees)(image)

def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    jitter_transform = transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue
    )
    return jitter_transform(image)

def grayscale(image):
    return transforms.functional.to_grayscale(image)

def affine_transform(image, angle=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10):
    transform = transforms.RandomAffine(
        degrees=angle,
        translate=translate,
        scale=scale,
        shear=shear
    )
    return transform(image)

def cutout(image, size=16):
    width, height = image.size
    left = random.randint(0, width - size)
    top = random.randint(0, height - size)
    right = left + size
    bottom = top + size
    image.paste((0, 0, 0), (left, top, right, bottom))
    return image

def sharpness(image, factor=2.0):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)
    
def apply_augmentation(image, augmentation_functions):
    augmented_images = [image]
    for aug_function in augmentation_functions:
        augmented_image = aug_function(image)
        augmented_images.append(augmented_image)
    return augmented_images
