import tensorflow as tf


def conv_block(x, filters):
    x = tf.keras.layers.Conv3D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv3D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    return x


def decoder_block(x, skip, filters):
    x = tf.keras.layers.Conv3DTranspose(filters, kernel_size=2, strides=2, padding="same")(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def Model_3UNet(input_shape=(128, 128, 128, 4), num_classes=4):
    inputs = tf.keras.Input(input_shape)

    s1 = conv_block(inputs, 16)
    p1 = tf.keras.layers.MaxPool3D(2)(s1)
    s2 = conv_block(p1, 32)
    p2 = tf.keras.layers.MaxPool3D(2)(s2)
    s3 = conv_block(p2, 64)
    p3 = tf.keras.layers.MaxPool3D(2)(s3)
    s4 = conv_block(p3, 128)
    p4 = tf.keras.layers.MaxPool3D(2)(s4)

    b = conv_block(p4, 256)

    d1 = decoder_block(b, s4, 128)
    d2 = decoder_block(d1, s3, 64)
    d3 = decoder_block(d2, s2, 32)
    d4 = decoder_block(d3, s1, 16)

    outputs = tf.keras.layers.Conv3D(num_classes, kernel_size=1, activation="softmax")(d4)
    return tf.keras.Model(inputs, outputs, name="3D_U-Net")


callbacks = [
        tf.keras.callbacks.ModelCheckpoint("3D_UNet.keras", save_best_only=True, monitor="val_Dice", mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_Dice", mode="max", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_Loss", factor=0.5, patience=3, mode="min")
    ]
