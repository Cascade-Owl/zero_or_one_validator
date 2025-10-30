import os
import cv2
import numpy as np


def preprocess_dataset(data_src_path, img_size=28):
    dataset_path = data_src_path
    x = []
    y = []

    for index, class_folder in enumerate(sorted(os.listdir(dataset_path))):
        class_path = os.path.join(dataset_path, class_folder)
        if not os.path.isdir(class_path):
            continue
        
        for file in os.listdir(class_path):
            if file.lower().endswith((".jpeg", ".jpg", ".png")):
                img_path = os.path.join(class_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))
                img = img / 255.0
                img_flattened = img.flatten()
                x.append(img_flattened)
                y.append(index)

    x = np.array(x)
    y = np.array(y)
    y = y.reshape(-1, 1)

    return x, y


def preprocess_image(img_path, img_size=28):
    img_path = os.path.abspath(img_path)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found at: {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"File cannot be loaded from the path: {img_path}")

    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img_flattened = img.flatten()
    img_flattened = img_flattened.reshape(1, -1)

    return img_flattened
