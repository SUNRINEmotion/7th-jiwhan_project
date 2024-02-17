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
 ```

라벨링데이터(json 파일)들을 8 : 2 의 비율로 train,test를 나눠줍니다. 나눈 데이터들을 이미 드라이브에 만들어 놓은 traindata,testdata폴더에 복사합니다.


 ```python
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
 ```

이 데이터는 라벨링데이터 하나당 대응하는 하나의 jpg 파일이 있고 json 파일에는 그 jpg 파일의 다양한 정보가 담겨있습니다. 이 코드에서는 json 파일에 대응하는 jpg 파일을 불러오고 그 jpg 파일의 정보가 담겨있는 json 파일의 damage 항목의 값을 추출하여 레이블로 사용합니다.
 ```python
print(f"레이블의 종류: {list(labels)}")
                  if label == "Scratched":
                        target_directory = "/content/drive/MyDrive/JH/damage/test/Scratched"
                    elif label == "Breakage":
                        target_directory = "/content/drive/MyDrive/JH/damage/test/Breakage"
                    elif label == "Separated":
                        target_directory = "/content/drive/MyDrive/JH/damage/test/Separated"
                    elif label == "Crushed":
                        target_directory = "/content/drive/MyDrive/JH/damage/test/Crushed"
                    else:
                        print(f"Unknown label: {label}. Skipping file: {json_file_name}")
                        continue
 ```
레이블의 종류를 확인해 드라이브에 레이블마다 폴더를 만들고 이미지의 레이블에 맞는 폴더에 이미지를 복사해줍니다.

 ```python
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/MyDrive/JH/damage/train',
    image_size=(224, 224),
    batch_size=64,
    seed=42,
    color_mode='rgb'
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/MyDrive/JH/damage/test',
    image_size=(224, 224),
    batch_size=64,
    seed=42,
    color_mode='rgb'
)

def preprocess_func(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label

train_ds = train_ds.map(preprocess_func)
test_ds = test_ds.map(preprocess_func)

def preprocess_funcc(image, label):
    label = tf.one_hot(label, depth=4)
    return image, label

train_ds = train_ds.map(preprocess_funcc)
test_ds = test_ds.map(preprocess_funcc)

 ```
모델 학습 부분에서 resnet50을 사용하기 위해 이미지를 224,224에 rgb로 바꿔주고 정규화와 원-핫 인코딩을 적용해줍니다.

 ```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(4, activation='softmax'))
model.summary()

model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])

def train_model(model, train_data, val_data, epochs):
    # 모델 학습
    history = model.fit(train_data, validation_data=val_data, epochs=epochs, verbose=1)
    return history

history = train_model(model, train_ds, test_ds, epochs=80)
```
모델 구성과 학습 부분입니다. resnet50으로 모델을 구성하며 학습된 가중치를 사용하였고 활성화 함수로는 relu함수, softmax함수를 사용하며 epochs는 80으로 모델을 실행시켰습니다.

 ```python
from tensorflow.keras.preprocessing import image

damage_labels = {0: 'Breakage', 1: 'Crushed', 2: 'Scratched', 3: 'Separated'}

def predict_damage(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array /= 255.0

    # 예측
    predictions = model.predict(img_array)[0]
    for i, score in enumerate(predictions):
        print(f"{damage_labels[i]}: {score*100:.2f}%")  # 각 손상 유형의 확률을 출력

    damage_idx = np.argmax(predictions)  # 가장 높은 확률을 가진 인덱스
    damage = damage_labels[damage_idx]

    return damage

img_path = '/content/drive/MyDrive/JH/996C0D3F5D785FDD03.jpg'
predicted_damage = predict_damage(img_path, model)
print(f"이미지의 손상: {predicted_damage}")
```
이미지를 넣으면 이미지를 인식하기 좋게 224,224로 만들어주고 정규화를 해줍니다. 그리고 앞서 학습한 모델을 이용하여 자동차 사진 이미지의 사고 유형을 인식하고 각 레이블의 확률을 출력하며 마지막에 가장 확률이 높은 레이블을 출력합니다.

![1378346_1187445_4241](https://github.com/SUNRINEmotion/7th-jiwhan_project/assets/79753445/4a0f52bb-543d-405e-8532-75a45ac39c3f)


#### 개선 방안
처음에는 데이터의 샘플데이터가 아니라 약 50gb의 원본 데이터를 이용해 학습을 할려했지만 너무 큰 데이터의 크기는 코랩 환경에서는 무리였습니다. 그래서 다음에는 코랩이 아닌 다른 환경에서 더 크고 자세한 데이터들을 이용하여 인공지능 개발을 하고 싶습니다.
