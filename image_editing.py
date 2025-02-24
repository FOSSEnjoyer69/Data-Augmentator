from pathlib import Path

import os
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from random import randint

def augment_dataset(dataset_folder_path:str, iterations:int=1):
    dataset_folder_path = Path(dataset_folder_path)
    image_file_paths = list(dataset_folder_path.glob("*.jpg")) + list(dataset_folder_path.glob("*.png")) + list(dataset_folder_path.glob("*.jpeg"))

    for image_file_path in image_file_paths:
        image = np.array(Image.open(image_file_path))
        images = create_random_variations(image, iterations)
        for i in range(len(images)):
            cv2.imwrite(f"Output/{str(randint(0, 12312435))}.jpg", images[i])


def create_random_variations(image:np.ndarray, iterations:int=1):
    outputs = []

    outputs.append(cv2.cvtColor(tf.image.random_flip_left_right(image).numpy(), cv2.COLOR_BGR2RGB))
    outputs.append(cv2.cvtColor(tf.image.random_flip_up_down(image).numpy(), cv2.COLOR_BGR2RGB))

    for i in range(iterations):
        outputs.append(cv2.cvtColor(tf.image.random_brightness(image, 1).numpy(), cv2.COLOR_BGR2RGB))
        outputs.append(cv2.cvtColor(tf.image.random_contrast(image, lower=0, upper=1).numpy(), cv2.COLOR_BGR2RGB))
        outputs.append(cv2.cvtColor(tf.image.random_hue(image, 0.2).numpy(), cv2.COLOR_BGR2RGB))
        outputs.append(cv2.cvtColor(tf.image.random_jpeg_quality(image, min_jpeg_quality=0, max_jpeg_quality=100).numpy(), cv2.COLOR_BGR2RGB))
        outputs.append(cv2.cvtColor(tf.image.random_saturation(image, lower=0, upper=1).numpy(), cv2.COLOR_BGR2RGB))
    return outputs
