import os
import cv2
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_dataset_structure(root_dir):
    for split in ["train", "validation", "test"]:
        os.makedirs(os.path.join(root_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root_dir, "masks", split), exist_ok=True)

def generate_mask(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([85, 50, 50])
    upper_blue = np.array([135, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def augment_image(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=10, p=0.5),
    ])
    augmented = transform(image=image, mask=mask)
    return augmented["image"], augmented["mask"]

def process_dataset(dataset_folder, output_folder, img_size=(512, 512), test_size=0.2, val_size=0.1):
    create_dataset_structure(output_folder)

    image_paths = [os.path.join(dataset_folder, img) for img in os.listdir(dataset_folder) if img.endswith((".jpg", ".png"))]
    train_imgs, test_imgs = train_test_split(image_paths, test_size=test_size, random_state=42)
    train_imgs, val_imgs = train_test_split(train_imgs, test_size=val_size, random_state=42)

    splits = {"train": train_imgs, "validation": val_imgs, "test": test_imgs}

    for split, img_list in splits.items():
        for img_path in tqdm(img_list, desc=f"Processing {split} images"):
            img_name = os.path.basename(img_path)
            image = cv2.imread(img_path)
            image = cv2.resize(image, img_size)
            mask = generate_mask(img_path)
            mask = cv2.resize(mask, img_size)

            if split == "train":
                image, mask = augment_image(image, mask)
            
            cv2.imwrite(os.path.join(output_folder, "images", split, img_name), image)
            cv2.imwrite(os.path.join(output_folder, "masks", split, img_name), mask)

if __name__ == "__main__":
    dataset_folder = "Water Bodies Dataset/Images"
    output_folder = "processed_dataset"
    process_dataset(dataset_folder, output_folder)