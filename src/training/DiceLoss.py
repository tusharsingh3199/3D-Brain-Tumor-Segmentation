import tensorflow as tf


def Dice(y_true, y_pred):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=4)
    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=4)

    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])

    return tf.reduce_mean((2. * intersection + smooth) / (union + smooth))


def Loss(y_true, y_pred):
    ce = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=4)
    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=4)

    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])

    dice_loss = 1 - tf.reduce_mean((2. * intersection + smooth) / (union + smooth))
    return dice_loss + ce


callbacks = [
    tf.keras.callbacks.ModelCheckpoint("3D_UNet.keras", save_best_only=True, monitor="val_Dice", mode="max"),
    tf.keras.callbacks.EarlyStopping(monitor="val_Dice", mode="max", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_Loss", factor=0.5, patience=3, mode="min")
]
