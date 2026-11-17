import tensorflow as tf
from tensorflow.keras import layers, Model


def build_baseline(input_shape=(224, 224, 5)):
    inp = layers.Input(input_shape)
    x = inp
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        if filters >= 128:
            x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out, name="baseline_cnn")


def build_efficientnet(input_shape=(224, 224, 5), weights="imagenet"):
    inp = layers.Input(input_shape, name="multispectral_input")
    # Trainable spectral adapter maps five satellite channels to ImageNet's 3-channel stem.
    x = layers.Conv2D(3, 1, padding="same", use_bias=False, name="spectral_adapter")(inp)
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=(224, 224, 3)
    )
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inp, out, name="efficientnetb0_multispectral")
    return model, base
