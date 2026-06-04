import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 讀取已訓練模型
model = load_model(
    "/content/drive/MyDrive/ML/final/pneumonia_model.h5"
)

# test 資料夾
test_dir = "/content/drive/MyDrive/ML/final/chest_xray/test"

# 類別
classes = ["NORMAL", "PNEUMONIA"]

# 隨機選擇 NORMAL 或 PNEUMONIA
random_class = random.choice(classes)

# 該類別資料夾
class_dir = os.path.join(test_dir, random_class)

# 隨機選一張圖片
random_image = random.choice(os.listdir(class_dir))

# 圖片完整路徑
img_path = os.path.join(class_dir, random_image)

# 載入圖片
img = image.load_img(img_path, target_size=(150,150))

# 轉 numpy
img_array = image.img_to_array(img)

# 正規化
img_array = img_array / 255.0

# 增加 batch 維度
img_array = np.expand_dims(img_array, axis=0)

# AI 預測
prediction = model.predict(img_array)

# prediction 會是 0~1
score = prediction[0][0]

# 判斷結果
if score > 0.5:
    result = "PNEUMONIA"
    confidence = score * 100
else:
    result = "NORMAL"
    confidence = (1 - score) * 100

# 顯示圖片
plt.imshow(img)

# 左上角文字
plt.title(
    f"Prediction: {result}\nConfidence: {confidence:.2f}%"
)

plt.axis("off")

plt.show()

# 印出真實答案
print("Actual:", random_class)
print("Prediction:", result)
print(f"Confidence: {confidence:.2f}%")