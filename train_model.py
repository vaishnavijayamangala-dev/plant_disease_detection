import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import os


train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 60


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

num_classes = train_data.num_classes

def se_block(input_tensor, ratio=8):

    filters = input_tensor.shape[-1]

    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters//ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)

    se = Reshape((1,1,filters))(se)

    x = Multiply()([input_tensor,se])

    return x



base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)


for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True

x = base_model.output


x = se_block(x)

x = GlobalAveragePooling2D()(x)


x = Dense(
    512,
    activation="swish",
    kernel_regularizer=l2(0.0005)
)(x)

x = BatchNormalization()(x)

x = Dropout(0.5)(x)


predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

model.summary()


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=5,
    min_lr=1e-6
)

checkpoint = ModelCheckpoint(
    "model/plant_disease_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)


history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr, checkpoint]
)
print("Training Completed")