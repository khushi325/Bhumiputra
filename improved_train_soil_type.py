"""
improved_train_soil_type.py

Better training flow:
 - computes class weights to handle imbalance
 - stronger augmentation
 - stage 1: train top classifier with base frozen
 - stage 2: unfreeze last layers and fine-tune with smaller LR
 - saves best model to 'soil_type_model_best.h5'
 - produces classification report and confusion matrix
 - copies misclassified validation images to 'misclassified/' for manual inspection
"""

import os, json, shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- USER SETTINGS ----------------
# Set these to your actual dataset paths
TRAIN_DIR = 'data/soil_images/Train'   # <-- adjust if needed
VAL_DIR   = 'data/soil_images/val'     # <-- adjust if needed

BACKBONE = 'MobileNetV2'   # 'MobileNetV2' or 'EfficientNetB0'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 8         # stage 1 (frozen base)
FINE_TUNE_EPOCHS = 8       # stage 2 (unfreeze)
FINE_TUNE_AT = 100         # unfreeze last N layers of base (use smaller number if base small)
BASE_LR = 1e-4
FINE_TUNE_LR = 1e-5
MODEL_BEST_PATH = 'soil_type_model_best.h5'
CLASS_MAP_PATH = 'class_indices.json'
MISCLASS_DIR = 'misclassified'
# ------------------------------------------------

# sanity checks
if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
    raise FileNotFoundError(f"Could not find dataset folders. Check TRAIN_DIR and VAL_DIR.\nTRAIN_DIR={TRAIN_DIR}\nVAL_DIR={VAL_DIR}")

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=(0.7,1.3),
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False   # important: keep order for evaluation / misclassified mapping
)

num_classes = train_gen.num_classes
print(f"Detected {num_classes} classes: {train_gen.class_indices}")

# compute class weights to penalize imbalance
y_train = train_gen.classes
unique_classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
class_weights_dict = {int(k): float(v) for k,v in zip(unique_classes, class_weights)}
print("Class weights:", class_weights_dict)

# save class indices for later use (streamlit)
with open(CLASS_MAP_PATH, 'w') as f:
    json.dump(train_gen.class_indices, f)

# Choose backbone
if BACKBONE == 'MobileNetV2':
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
elif BACKBONE == 'EfficientNetB0':
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
else:
    raise ValueError("BACKBONE must be 'MobileNetV2' or 'EfficientNetB0'")

# Build head
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base.input, outputs=outputs)

# Stage 1: freeze base
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=BASE_LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("=== Stage 1: training top layers with base frozen ===")
callbacks = [
    ModelCheckpoint(MODEL_BEST_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# Stage 2: fine-tune - unfreeze top layers of base
print("\n=== Stage 2: fine-tuning top layers of the base model ===")
# unfreeze from layer index: set last N layers trainable
if FINE_TUNE_AT < 0:
    # unfreeze all
    for layer in base.layers:
        layer.trainable = True
else:
    # freeze first layers, unfreeze last FINE_TUNE_AT layers
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= (len(base.layers) - FINE_TUNE_AT)

# recompile with lower LR
model.compile(optimizer=Adam(learning_rate=FINE_TUNE_LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history1.epoch[-1] if hasattr(history1, 'epoch') and len(history1.epoch)>0 else 0,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# final save (best already saved by checkpoint; save final too)
model.save('soil_type_model_final.h5')
print("Saved final model to soil_type_model_final.h5")

# Evaluate on validation set
val_gen.reset()
y_true = val_gen.classes
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

index_to_label = {v:k for k,v in train_gen.class_indices.items()}
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=[index_to_label[i] for i in range(num_classes)]))

# Confusion matrix
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=[index_to_label[i] for i in range(num_classes)],
                yticklabels=[index_to_label[i] for i in range(num_classes)],
                cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")
except Exception:
    print("matplotlib not available — skipping confusion matrix plot")

# Save misclassified examples for manual inspection
os.makedirs(MISCLASS_DIR, exist_ok=True)
# clear previous
for f in os.listdir(MISCLASS_DIR):
    fp = os.path.join(MISCLASS_DIR, f)
    if os.path.isfile(fp):
        os.remove(fp)

# val_gen.filenames are relative to VAL_DIR root (e.g., 'Alluvial soil/img1.jpg')
for i, rel_path in enumerate(val_gen.filenames):
    true_idx = y_true[i]
    pred_idx = y_pred[i]
    if true_idx != pred_idx:
        src = os.path.join(VAL_DIR, rel_path)
        # name file with true_pred_originalname.jpg
        safe_name = rel_path.replace(os.sep, "__").replace(" ", "_")
        dst = os.path.join(MISCLASS_DIR, f"true{true_idx}_pred{pred_idx}__{safe_name}")
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            # some generators may have windows path differences, attempt alternative
            alt_src = os.path.join(VAL_DIR, rel_path.replace("/", os.sep))
            try:
                shutil.copy2(alt_src, dst)
            except Exception:
                pass

print(f"Saved misclassified images to folder: {MISCLASS_DIR}")
print("Done.")
