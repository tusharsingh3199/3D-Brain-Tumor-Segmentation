def decoder_block(x, skip, filters):
    x = tf.keras.layers.Conv3DTranspose(filters, kernel_size=2, strides=2, padding="same")(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x
