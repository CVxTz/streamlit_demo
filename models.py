from tensorflow.keras.layers import (
    Dense,
    Input,
    GlobalMaxPooling2D,
    Convolution2D,
    MaxPooling2D
)
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def get_model_cnn(
        input_shape=(None, None, 1),
        n_classes=10,
):
    inputs = Input(input_shape)

    x = Convolution2D(64, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling2D(pool_size=2)(x)

    x = Convolution2D(64, kernel_size=3, activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Convolution2D(64, kernel_size=3, activation="relu")(x)

    x = GlobalMaxPooling2D()(x)

    out1 = Dense(2, activation="linear")(x)
    out = Dense(50, activation="relu")(out1)
    out = Dense(n_classes, activation="softmax")(out)

    model = Model(inputs, out)
    model.compile(optimizer=Adam(0.0001), loss=categorical_crossentropy, metrics=["acc"])

    model_aux = Model(inputs, out1)
    model_aux.compile(optimizer=Adam(0.0001), loss=categorical_crossentropy, metrics=["acc"])

    model.summary()

    return model


def get_model_mlp(
        input_shape=(784,), n_classes=10,
):
    inputs = Input(input_shape)
    x = Dense(32, activation="relu")(inputs)
    x = Dense(32, activation="relu")(x)

    out1 = Dense(2, activation="linear")(x)
    out = Dense(50, activation="relu")(out1)
    out = Dense(n_classes, activation="softmax")(out)

    model = Model(inputs, out)

    model.compile(
        optimizer=Adam(0.0001), loss=categorical_crossentropy, metrics=["acc"]
    )

    model_aux = Model(inputs, out1)
    model_aux.compile(optimizer=Adam(0.0001), loss=categorical_crossentropy, metrics=["acc"])

    model.summary()

    return model
