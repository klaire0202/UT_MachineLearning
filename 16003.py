import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 路徑（已幫你改好）
train_dir = "/content/drive/MyDrive/ML/final/chest_xray/train"
test_dir = "/content/drive/MyDrive/ML/final/chest_xray/test"

# 前處理（把像素縮到 0~1）
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

# 載入資料
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

# 建立模型（最簡 CNN）
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 編譯
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 訓練（先跑 x 次就好，測試用）
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator
)

# 儲存模型
model.save("/content/drive/MyDrive/ML/final/pneumonia_model.h5")

print("模型已儲存")
