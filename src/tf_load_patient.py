def tf_load_patient(patient_dir):
    image, mask = tf.py_function(load_patient, [patient_dir], [tf.float32, tf.int32])
    image.set_shape((128, 128, 128, 4))
    mask.set_shape((128, 128, 128))
    return image, mask