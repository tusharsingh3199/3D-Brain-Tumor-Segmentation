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
    return dice_loss + c