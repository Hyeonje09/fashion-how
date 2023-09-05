import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
from augmentation import *

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 증강 함수들
augmentation_functions = [
    gaussian_blur,
    flip,
    plus_rotate,
    minus_rotate,
]

# 경로 설정
csv_path = './Dataset/info_etri20_emotion_train.csv'
base_image_path = './Dataset/Train/'

# 기존 CSV 파일 읽기
data = pd.read_csv(csv_path)

new_rows = []

# 데이터 순회 및 수정
for idx, row in tqdm(data.iterrows(), total=len(data)):
    image_name = row['image_name']
    image_path = os.path.join(base_image_path, image_name)
    
    # 이미지 읽기
    image = Image.open(image_path)
    
    # 이미지를 Pillow 형식으로 변환
    pil_image = transforms.ToPILImage()(transforms.ToTensor()(image))
    
    augmented_images = [pil_image]  # 원본 이미지를 먼저 추가
    
    for aug_function in augmentation_functions:
        augmented_image = aug_function(pil_image)
        augmented_images.append(augmented_image)
    
    for aug_idx, augmented_image in enumerate(augmented_images[1:], start=1):  # 첫 번째 이미지(원본)를 제외하고 순회
        # 이미지를 Tensor로 변환
        augmented_image_tensor = transforms.ToTensor()(augmented_image).to(device)
        
        # 증강된 이미지 저장 경로 설정
        folder_name = os.path.dirname(image_name)
        image_name_without_folders = os.path.basename(image_name)
        aug_name = augmentation_functions[aug_idx - 1].__name__
        augmented_image_name = f"{image_name_without_folders[:-4]}_{aug_name}_augmented.jpg"
        augmented_folder_path = os.path.join(base_image_path, folder_name)
        augmented_image_path = os.path.normpath(os.path.join(augmented_folder_path, augmented_image_name))
        
        # 증강된 이미지 저장
        os.makedirs(augmented_folder_path, exist_ok=True)
        augmented_image.save(augmented_image_path, format="JPEG")  # JPG 형식으로 저장
        
        # 증강된 이미지 정보 생성
        new_row = row.copy()
        new_row['image_name'] = os.path.join(folder_name, augmented_image_name)
        new_rows.append(new_row)

# 새로운 데이터로 DataFrame 생성
new_data = pd.DataFrame(new_rows)

# 기존 CSV 파일 업데이트
updated_data = pd.concat([data, new_data], ignore_index=True)
updated_data.to_csv(csv_path, index=False)
