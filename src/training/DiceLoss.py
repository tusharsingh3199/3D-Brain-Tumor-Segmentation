import tensorflow as tf
from configs.config import CLASS_WEIGHTS


def Dice_Class(y_true, y_pred, class_idx):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=4)
    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=4)
    yt = y_true[..., class_idx]
    yp = y_pred[..., class_idx]

    smooth = 1e-6
    intersection = tf.reduce_sum(yt * yp, axis=[1,2,3])
    union = tf.reduce_sum(yt + yp, axis=[1,2,3])

    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)


def Dice_NCR(y_true, y_pred):
    return Dice_Class(y_true, y_pred, 1)


def Dice_ED(y_true, y_pred):
    return Dice_Class(y_true, y_pred, 2)


def Dice_ET(y_true, y_pred):
    return Dice_Class(y_true, y_pred, 3)


def Dice(y_true, y_pred):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=4)
    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=4)
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])

    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)


def Loss(y_true, y_pred):
    ce = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=4)
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1. - 1e-6)

    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])

    dice = (2. * intersection + smooth) / (union + smooth)
    weighted_dice = tf.reduce_sum(dice * CLASS_WEIGHTS) / tf.reduce_sum(CLASS_WEIGHTS)
    dice_loss = 1.0 - weighted_dice
    return dice_loss + ce

