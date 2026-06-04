import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# 載入已訓練模型
# =========================
model = load_model(
    "/content/drive/MyDrive/ML/final/pneumonia_model.h5"
)

# =========================
# 測試資料夾路徑
# =========================
test_dir = "/content/drive/MyDrive/ML/final/chest_xray/test"

# =========================
# 建立輸出資料夾
# =========================
output_dir = "demo"

os.makedirs(output_dir, exist_ok=True)

# =========================
# 類別名稱
# =========================
classes = ["NORMAL", "PNEUMONIA"]

# =========================
# 收集所有測試圖片
# =========================
all_images = []

for class_name in classes:

    class_dir = os.path.join(test_dir, class_name)

    for img_name in os.listdir(class_dir):

        img_path = os.path.join(class_dir, img_name)

        all_images.append((img_path, class_name))

# =========================
# 隨機抽 10 張圖片
# =========================
random_images = random.sample(all_images, 10)

# =========================
# 開始預測
# =========================
for i, (img_path, actual_class) in enumerate(random_images):

    # 讀取圖片
    img = image.load_img(img_path, target_size=(150,150))

    # 轉成 numpy array
    img_array = image.img_to_array(img)

    # 正規化
    img_array = img_array / 255.0

    # 增加 batch 維度
    img_array = np.expand_dims(img_array, axis=0)

    # 模型預測
    prediction = model.predict(img_array)

    score = prediction[0][0]

    # 判斷類別
    if score > 0.5:

        predicted_class = "PNEUMONIA"
        confidence = score * 100

    else:

        predicted_class = "NORMAL"
        confidence = (1 - score) * 100

    # =========================
    # 顯示圖片
    # =========================
    plt.figure(figsize=(5,5))

    plt.imshow(img)

    # 標題
    plt.title(
        f"Actual: {actual_class}\n"
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.2f}%"
    )

    plt.tight_layout()

    plt.axis("off")

    # =========================
    # 儲存圖片
    # =========================
    save_path = os.path.join(
        output_dir,
        f"demo_{i+1}.png"
    )

    plt.savefig(save_path)

    print(f"已儲存：{save_path}")

    plt.close()

# =========================
# 完成訊息
# =========================
print("\nDemo 完成！")
print("圖片已儲存到 demo 資料夾")
