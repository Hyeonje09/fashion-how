from torchvision import transforms
from PIL import ImageFilter

def gaussian_blur(image, kernel_size=5):
    image_pil = image.filter(ImageFilter.GaussianBlur(kernel_size))
    return image_pil

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