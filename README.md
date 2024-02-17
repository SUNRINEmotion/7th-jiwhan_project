# 자동차 사고 종류 인식


#### 프로젝트 동기

만약 자동차 사고가 일어났을 때 사고의 종류에 따라 적절하고 빠르게 대응하면 많은 시간과 돈을 아낄 수 있겠다는 생각이 들어 이 프로젝트를 시작하게 되었습니다.

#### 프로젝트 설명

이 프로젝트에 사용된 데이터는 ai hub사이트의 차량 파손 이미지 데이터의 샘플 데이터입니다.

 ``` python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
from google.colab import drive
import os
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from shutil import copyfile

drive.mount('/content/drive')
 ```

resnet50을 포함한 프로젝트에 필요한 주요한 라이브러리들을 호출해 주고 CNN에 필요한 Flatten, Dropout, GlobalAveragePooling2D등을 불러와 줍니다. 또한 코랩 환경에서 드라이브를 사용하기 위해 드라이브를 불러와줍니다.


 ``` python
import os
import shutil
from sklearn.model_selection import train_test_split

# 경로 설정
directory_path = "/content/drive/MyDrive/JH/라벨링데이터/TL_damage/damage"

# 모든 파일 목록 가져오기
file_list = os.listdir(directory_path)

# 파일 목록을 8:2 비율로 나누기
train_files, test_files = train_test_split(file_list, test_size=0.2, random_state=42)

# 훈련 데이터와 테스트 데이터를 저장할 경로
train_directory_path = "/content/drive/MyDrive/JH/라벨링데이터/TL_damage/traindata"
test_directory_path = "/content/drive/MyDrive/JH/라벨링데이터/TL_damage/testdata"

# 훈련 데이터 파일을 해당 디렉토리로 복사
for file_name in train_files:
    shutil.copy(os.path.join(directory_path, file_name), 
                os.path.join(train_directory_path, file_name))

# 테스트 데이터 파일을 해당 디렉토리로 복사
for file_name in test_files:
    shutil.copy(os.path.join(directory_path, file_name), 
                os.path.join(test_directory_path, file_name))
'''

라벨링데이터(json파일)들을 8 : 2 의 비율로 train,test를 나눠줍니다. 나눈 데이터들을 이미 드라이브에 만들어 놓은 traindata,testdata폴더에 복사합니다.


''' python
# 폴더 경로 설정
json_directory = "/content/drive/MyDrive/JH/라벨링데이터/TL_damage/traindata"
image_directory = "/content/drive/MyDrive/JH/원천데이터/TS_damage/damage"

# 이미지와 레이블을 저장할 리스트
images = []
labels = []

for json_file_name in os.listdir(json_directory):
    try:
        if json_file_name.endswith('.json'):
            json_file_path = os.path.join(json_directory, json_file_name)

            # JSON 파일 읽기
            with open(json_file_path, 'r') as json_file:
                json_data = json.load(json_file)

                # 이미지 파일 경로 생성
                image_file_name = json_data["images"]["file_name"]
                image_file_path = os.path.join(image_directory, image_file_name)

                # 이미지 파일이 존재하는지 확인
                if os.path.exists(image_file_path):

                    # 이미지 로드 및 전처리
                    image = load_img(image_file_path, target_size=(224, 224))
                    image_array = img_to_array(image)

                    # 레이블 추출
                    label = json_data["annotations"][0]["damage"]

                    # 이미지와 레이블 저장
                    images.append(image_array)
                    labels.append(label)
'''

이 데이터는 라벨링데이터 하나 당 대응하는 하나의 jpg파일이 있고 json파일에는 그 jpg파일의 다양한 정보가 담겨있습니다. 이 코드에서는 json파일에 대응하는 jpg파일을 불러오고 그 jpg파일의 정보가 담겨있는 json파일의 damage항목의 값을 추출하여 레이블로 사용합니다.


