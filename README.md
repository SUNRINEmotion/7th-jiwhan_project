### 자동차 사고 종류 인식

# 프로젝트 동기

만약 자동차 사고가 일어났을 때 사고의 종류에 따라 적절하고 빠르게 대응하면 많은 시간과 돈을 아낄 수 있겠다는 생각이 들어 이 프로젝트를 시작하게 되었습니다.

# 프로젝트 설명
''' python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
from google.colab import drive


drive.mount('/content/drive')
'''
