import numpy as np
import tensorflow as tf

def rounder(nb, digit_to_keep):
    mult = 10 ** digit_to_keep
    return np.round(nb*mult)/mult

def normalize_eigenvecs(eigenvecs):
    vector_norm = tf.math.sqrt(
                    tf.reduce_sum(tf.reduce_sum(eigenvecs**2, 3), 1)
                )
    vector_norm = tf.repeat(vector_norm[:, tf.newaxis, :], 64, axis=1)[
        :, :, :, tf.newaxis
    ]
    vector_norm = tf.concat([vector_norm, vector_norm], axis=3)  # real, imag
    return tf.divide(eigenvecs, vector_norm)

