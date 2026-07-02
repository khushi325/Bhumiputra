import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

train_dir = 'data/plant_village/train'
val_dir = 'data/plant_village/val'

img_size = (224,224)
batch_size = 16

if not os.path.exists(train_dir):
    raise SystemExit('Missing training folder: data/plant_village/train. Create folders per README.')

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
val_gen = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0],img_size[1],3))
base.trainable = False
x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs=base.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_gen, epochs=5, validation_data=val_gen)
model.save('crop_model.h5')
print('Saved crop_model.h5')