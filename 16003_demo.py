import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model(
    "/content/drive/MyDrive/ML/final/pneumonia_model.h5"
)

test_dir = "/content/drive/MyDrive/ML/final/chest_xray/test"

output_dir = "demo"

os.makedirs(output_dir, exist_ok=True)

classes = ["NORMAL", "PNEUMONIA"]

all_images = []

for class_name in classes:

    class_dir = os.path.join(test_dir, class_name)

    for img_name in os.listdir(class_dir):

        img_path = os.path.join(class_dir, img_name)

        all_images.append((img_path, class_name))

random_images = random.sample(all_images, 10)

for i, (img_path, actual_class) in enumerate(random_images):

    img = image.load_img(img_path, target_size=(150,150))

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    score = prediction[0][0]

    if score > 0.5:

        predicted_class = "PNEUMONIA"
        confidence = score * 100

    else:

        predicted_class = "NORMAL"
        confidence = (1 - score) * 100

    plt.figure(figsize=(5,5))

    plt.imshow(img)

    plt.title(
        f"Actual: {actual_class}\n"
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.2f}%"
    )

    plt.tight_layout()

    plt.axis("off")

    save_path = os.path.join(
        output_dir,
        f"demo_{i+1}.png"
    )

    plt.savefig(save_path)

    print(f"已儲存：{save_path}")

    plt.close()

print("\nDemo 完成！")
print("圖片已儲存到 demo 資料夾")
