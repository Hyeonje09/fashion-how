from torchvision import transforms
from PIL import ImageFilter

def gaussian_blur(image, kernel_size=3):
    image_pil = image.filter(ImageFilter.GaussianBlur(kernel_size))
    return image_pil

def flip(image):
    return transforms.functional.hflip(image)

def random_rotate(image, degrees=20):
    return transforms.RandomRotation(degrees)(image)

def apply_augmentation(image, augmentation_functions):
    augmented_images = [image]
    for aug_function in augmentation_functions:
        augmented_image = aug_function(image)
        augmented_images.append(augmented_image)
    return augmented_images
