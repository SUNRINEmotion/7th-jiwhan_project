# 자동차 사고 종류 인식


#### 프로젝트 동기

만약 자동차 사고가 일어났을 때 사고의 종류에 따라 적절하고 빠르게 대응하면 많은 시간과 돈을 아낄 수 있겠다는 생각이 들어 이 프로젝트를 시작하게 되었습니다.

#### 프로젝트 설명
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

