import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from mtcnn import MTCNN

# ==============================
# PATHS
# ==============================
raw_dataset_root = r"C:\\Mrithika\\kadir\\ResNetFinal\\dataset_classification"
train_dir_raw = os.path.join(raw_dataset_root, "train")
val_dir_raw = os.path.join(raw_dataset_root, "valid")

faces_dataset_root = r"C:\\Mrithika\\kadir\\ResNetFinal\\dataset_faces"
train_dir = os.path.join(faces_dataset_root, "train")
val_dir = os.path.join(faces_dataset_root, "valid")

model_path = r"C:\\Mrithika\\kadir\\ResNetFinal\\mobilenet_malnutrition_optimizedd.h5"

# ==============================
# STEP 1: FACE CROPPING
# ==============================
detector = MTCNN()

def crop_faces_from_dir(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for class_name in os.listdir(src_dir):
        class_src = os.path.join(src_dir, class_name)
        class_dest = os.path.join(dest_dir, class_name)
        os.makedirs(class_dest, exist_ok=True)

        for fname in os.listdir(class_src):
            img_path = os.path.join(class_src, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)
            if len(faces) == 0:
                continue
            x, y, w, h = faces[0]['box']
            face = img_rgb[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            save_path = os.path.join(class_dest, fname)
            cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

print("Cropping train faces...")
crop_faces_from_dir(train_dir_raw, train_dir)
print("Cropping validation faces...")
crop_faces_from_dir(val_dir_raw, val_dir)
print("✅ Cropped faces dataset ready!")

# ==============================
# STEP 2: DATA GENERATORS
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.4,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    channel_shift_range=30.0,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# ==============================
# STEP 3: MODEL BUILDING
# ==============================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Handle class imbalance
counts = train_gen.classes
unique, counts = np.unique(counts, return_counts=True)
class_weights = None
if len(counts) == 2 and counts[0] != counts[1]:
    class_weights = {0: counts[1]/counts[0], 1: 1.0}

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)
]

# ==============================
# STEP 4: TRAINING (HEAD)
# ==============================
print("\n🚀 Starting initial training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weights
)

# ==============================
# STEP 5: FINE-TUNING
# ==============================
for layer in base_model.layers[-100:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🔧 Starting fine-tuning...")
history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weights
)

model.save(model_path)
print(f"\n✅ Model saved as {model_path}")

val_acc_head = history.history['val_accuracy'][-1]
val_acc_ft = history_ft.history['val_accuracy'][-1]
print(f"Validation Accuracy after head training: {val_acc_head*100:.2f}%")
print(f"Validation Accuracy after fine-tuning: {val_acc_ft*100:.2f}%")

# ==============================
# STEP 6: PREDICTION ON MULTIPLE IMAGES
# ==============================
print("\n📸 Running predictions...")
model_pathh="C:\\Mrithika\\kadir\\ResNetFinal\\mobilenet_malnutrition_optimized.h5"
model = load_model(model_pathh)
print("✅ Model loaded successfully!")

test_image_paths = [
    r"C:\\Mrithika\\kadir\\ResNetFinal\\image1.jpg",
    r"C:\\Mrithika\\kadir\\ResNetFinal\\image2.png",
    r"C:\\Mrithika\\kadir\\ResNetFinal\\image3.jpg",
    r"C:\\Mrithika\\kadir\\ResNetFinal\\image4.jpeg"
]

def predict_child(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No image found at {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    x = img_resized / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]

    classes = ['Healthy', 'Malnourished']
    predicted_class = classes[np.argmax(preds)]
    confidence = preds[np.argmax(preds)] * 100
    return predicted_class, confidence

for path in test_image_paths:
    pred_class, pred_conf = predict_child(path)
    print(f"🖼️ Image: {os.path.basename(path)} → {pred_class} ({pred_conf:.2f}% confidence)")
