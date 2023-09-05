import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from augmentation import *
from torchvision import transforms

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
    
    # 이미지를 Tensor로 변환하고 GPU로 이동
    image_tensor = transforms.ToTensor()(image).to(device) 
    
    augmented_images = apply_augmentation(image_tensor, augmentation_functions)
    
    for aug_idx, augmented_image in enumerate(augmented_images):
        augmented_image_3d = augmented_image.cpu()
        pil_image = transforms.ToPILImage()(augmented_image_3d)
        
        # 증강된 이미지 저장 경로 설정
        folder_name = os.path.dirname(image_name)
        image_name_without_folders = os.path.basename(image_name)
        aug_name = augmentation_functions[aug_idx].__name__
        augmented_image_name = f"{image_name_without_folders[:-4]}_{aug_name}_augmented.jpg"
        augmented_folder_path = os.path.join(base_image_path, folder_name)
        augmented_image_path = os.path.normpath(os.path.join(augmented_folder_path, augmented_image_name))
        
        # 증강된 이미지 저장
        os.makedirs(augmented_folder_path, exist_ok=True)
        pil_image.save(augmented_image_path)
        
        # 증강된 이미지 정보 생성
        new_row = row.copy()
        new_row['image_name'] = os.path.join(folder_name, augmented_image_name)
        new_rows.append(new_row)

# 새로운 데이터로 DataFrame 생성
new_data = pd.DataFrame(new_rows)

# 기존 CSV 파일 업데이트
updated_data = pd.concat([data, new_data], ignore_index=True)
updated_data.to_csv(csv_path, index=False)
