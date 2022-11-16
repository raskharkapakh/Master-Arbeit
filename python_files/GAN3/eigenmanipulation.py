import tensorflow as tf


def normalize_evals(evals):
    return evals/tf.reduce_max(evals)

def normalize_evecs(eigenvecs):
    vector_norm = tf.math.sqrt(
                    tf.reduce_sum(tf.reduce_sum(eigenvecs**2, 3), 1)
                )
    vector_norm = tf.repeat(vector_norm[:, tf.newaxis, :], 64, axis=1)[
        :, :, :, tf.newaxis
    ]
    vector_norm = tf.concat([vector_norm, vector_norm], axis=3)  # real, imag
    return tf.divide(eigenvecs, vector_norm)

