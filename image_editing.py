from pathlib import Path

import os
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
import cv2
from random import randint

from icecream import ic

def get_common_filenames(folders):
    """
    For each folder in the list (which should be absolute paths),
    build a mapping from filename (basename) to the list of absolute file paths
    where that file is found. Then return only those filenames that appear
    in every folder.
    """
    file_locations = {}
    for folder in folders:
        # Make sure folder is absolute
        abs_folder = os.path.abspath(folder)
        for f in os.listdir(abs_folder):
            abs_path = os.path.abspath(os.path.join(abs_folder, f))
            if os.path.isfile(abs_path):
                file_locations.setdefault(f, []).append(abs_path)
    
    # Only keep file names that appear in all folders
    common_files = {fname: paths for fname, paths in file_locations.items() if len(paths) == len(folders)}
    return common_files

def augment_dataset(dataset_folder_path: str, folder_type: str, iterations: int = 1):
    # Convert the dataset folder path to an absolute path
    dataset_folder_path = os.path.abspath(dataset_folder_path)
    
    if folder_type == "Single Folder":
        dataset_folder = Path(dataset_folder_path)
        # Use Path.glob on the absolute path
        image_file_paths = list(dataset_folder.glob("*.jpg")) + list(dataset_folder.glob("*.png")) + list(dataset_folder.glob("*.jpeg"))

        for image_file_path in image_file_paths:
            image = np.array(Image.open(image_file_path))
            images = create_random_variations(image, iterations)
            for img in images:
                # Write using an absolute path for the output file
                output_file = os.path.abspath(f"Output/{randint(0, 12312435)}.jpg")
                cv2.imwrite(output_file, img)
    elif folder_type == "Parent Of Multiple":
        # Build a list of absolute paths for sub-folders
        sub_folders = [
            os.path.abspath(os.path.join(dataset_folder_path, f))
            for f in os.listdir(dataset_folder_path)
            if os.path.isdir(os.path.join(dataset_folder_path, f))
        ]

        output_sub_folders = []

        for sub_folder in sub_folders:
            folder_path = f"Output/{sub_folder.split("/")[-1]}"
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

            output_sub_folders.append(folder_path)

        #Copy Original Files
        common = get_common_filenames(sub_folders)
        #print("Common filenames across sub-folders:")
        for file_name, paths in common.items():
            #print(f"File name: {file_name}")
            for path in paths:
                #print(f"  {path}")

                folder_name = path.split("/")[-2]
                file_name = path.split("/")[-1]#
                bgr_image = np.array(Image.open(path))
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                cv2.imwrite(f"Output/{folder_name}/{file_name}", rgb_image)

        #Augment Data
        file_names = os.listdir(output_sub_folders[0])

        for file_name in file_names:
            seed = randint(0, 9999999)
            for output_sub_folder in output_sub_folders:
                image = np.array(Image.open(f"{output_sub_folder}/{file_name}"))
                vartiations = create_random_variations(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1, seed)
                for y in range((len(vartiations))):
                    cv2.imwrite(f"{output_sub_folder}/{seed + y}.jpg", vartiations[y])

def create_random_variations(image: np.ndarray, iterations: int = 1, seed: int = None, enable_flip:bool=True):
    outputs = []
    # Use the base seed for the first two operations
    base_seed = seed if seed is not None else None
    
    if enable_flip:
        outputs.append(random_hor_flip(image, seed))
        outputs.append(random_ver_flip(image, seed))
        #outputs.append(cv2.cvtColor(tf.image.random_flip_left_right(image).numpy(), cv2.COLOR_BGR2RGB))
        #outputs.append(cv2.cvtColor(tf.image.random_flip_up_down(image).numpy(), cv2.COLOR_BGR2RGB))

    for i in range(iterations):
        # Offset the seed per iteration if provided, otherwise leave as None
        current_seed = None if seed is None else seed + i
        outputs.append(random_brightness(image, seed))
        outputs.append(random_contrast(image, seed))
        outputs.append(random_hue(image, seed))
        outputs.append(random_jpg_quality(image, seed))
        outputs.append(random_saturation(image, seed))
        #outputs.append(cv2.cvtColor(tf.image.random_brightness(image, max_delta=1, seed=current_seed).numpy(), cv2.COLOR_BGR2RGB))
        #outputs.append(cv2.cvtColor(tf.image.random_contrast(image, lower=0, upper=1, seed=current_seed).numpy(), cv2.COLOR_BGR2RGB))
        #outputs.append(cv2.cvtColor(tf.image.random_hue(image, max_delta=0.2, seed=current_seed).numpy(), cv2.COLOR_BGR2RGB))
        #outputs.append(cv2.cvtColor(tf.image.random_jpeg_quality(image, min_jpeg_quality=0, max_jpeg_quality=100, seed=current_seed).numpy(), cv2.COLOR_BGR2RGB))
        #outputs.append(cv2.cvtColor(tf.image.random_saturation(image, lower=0, upper=1, seed=current_seed).numpy(), cv2.COLOR_BGR2RGB))
    
    return outputs       

#def create_random_variations(image:np.ndarray, iterations:int=1):
#    outputs = []
#
#    outputs.append(cv2.cvtColor(tf.image.random_flip_left_right(image).numpy(), cv2.COLOR_BGR2RGB))
#    outputs.append(cv2.cvtColor(tf.image.random_flip_up_down(image).numpy(), cv2.COLOR_BGR2RGB))
#
#    for i in range(iterations):
#        outputs.append(cv2.cvtColor(tf.image.random_brightness(image, 1).numpy(), cv2.COLOR_BGR2RGB))
#        outputs.append(cv2.cvtColor(tf.image.random_contrast(image, lower=0, upper=1).numpy(), cv2.COLOR_BGR2RGB))
#        outputs.append(cv2.cvtColor(tf.image.random_hue(image, 0.2).numpy(), cv2.COLOR_BGR2RGB))
#        outputs.append(cv2.cvtColor(tf.image.random_jpeg_quality(image, min_jpeg_quality=0, max_jpeg_quality=100).numpy(), cv2.COLOR_BGR2RGB))
#        outputs.append(cv2.cvtColor(tf.image.random_saturation(image, lower=0, upper=1).numpy(), cv2.COLOR_BGR2RGB))
#    return outputs

def random_hor_flip(image: np.ndarray, seed: int) -> np.ndarray:
    # Create a random number generator with the provided seed for reproducibility
    rng = np.random.default_rng(seed)
    # Generate a random number; flip the image if the number is > 0.5
    if rng.random() > 0.5:
        # Flip the image horizontally
        return np.fliplr(image)
    # Return the original image if not flipped
    return image

import numpy as np

def random_ver_flip(image: np.ndarray, seed: int) -> np.ndarray:
    # Create a random number generator with the provided seed for reproducibility
    rng = np.random.default_rng(seed)
    # Generate a random number; flip the image vertically if the number is > 0.5
    if rng.random() > 0.5:
        # Flip the image vertically
        return np.flipud(image)
    # Return the original image if not flipped
    return image

def random_brightness(image: np.ndarray, seed: int) -> np.ndarray:
    # Create a random number generator with the provided seed for reproducibility
    rng = np.random.default_rng(seed)
    # Choose a brightness factor in a range (e.g., between 0.5 and 1.5)
    brightness_factor = rng.uniform(0.5, 1.5)
    
    # Adjust the image brightness by multiplying with the brightness factor.
    # If the image is of integer type, convert to float for calculation, then clip and convert back.
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) * brightness_factor
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)
    else:
        return image * brightness_factor

def random_contrast(image: np.ndarray, seed: int) -> np.ndarray:
    """
    Randomly adjust the contrast of an image.
    Contrast factor is chosen uniformly from [0.5, 1.5].
    """
    rng = np.random.default_rng(seed)
    factor = rng.uniform(0.5, 1.5)
    pil_image = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_image = enhancer.enhance(factor)
    return np.array(enhanced_image)

def random_hue(image: np.ndarray, seed: int) -> np.ndarray:
    """
    Randomly shift the hue of an image.
    The hue shift is chosen from [-0.1, 0.1] (scaled to 0-255).
    """
    rng = np.random.default_rng(seed)
    # Calculate a hue shift: e.g., [-25.5, 25.5]
    hue_shift = int(rng.uniform(-0.1, 0.1) * 255)
    
    pil_image = Image.fromarray(image)
    hsv_image = pil_image.convert('HSV')
    h, s, v = hsv_image.split()
    
    # Shift the hue channel with wrap-around
    h_np = np.array(h, dtype=np.uint8)
    h_np = ((h_np.astype(int) + hue_shift) % 256).astype(np.uint8)
    h = Image.fromarray(h_np, mode='L')
    
    new_hsv = Image.merge('HSV', (h, s, v))
    new_rgb = new_hsv.convert('RGB')
    return np.array(new_rgb)

def random_jpg_quality(image: np.ndarray, seed: int) -> np.ndarray:
    """
    Simulate JPEG compression by encoding and decoding the image in memory.
    The JPEG quality is randomly chosen as an integer in the range [30, 100].
    This function does not save the image to disk.
    """
    rng = np.random.default_rng(seed)
    quality = int(rng.integers(30, 101))
    
    # OpenCV expects images in BGR order.
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    # Encode the image as JPEG into an in-memory buffer.
    result, encimg = cv2.imencode('.jpg', bgr_image, encode_param)
    if not result:
        raise ValueError("JPEG encoding failed")
    
    # Decode the image back from the buffer.
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
    return rgb_image

def random_saturation(image: np.ndarray, seed: int) -> np.ndarray:
    """
    Randomly adjust the saturation of an image.
    Saturation factor is chosen uniformly from [0.5, 1.5].
    """
    rng = np.random.default_rng(seed)
    factor = rng.uniform(0.5, 1.5)
    pil_image = Image.fromarray(image)
    enhancer = ImageEnhance.Color(pil_image)
    enhanced_image = enhancer.enhance(factor)
    return np.array(enhanced_image)