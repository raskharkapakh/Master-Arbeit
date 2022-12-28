import tensorflow as tf
import numpy as np

def levelify(evals):
    return 10*np.log10(evals)

def unlevelify(evals_dB):
    return 10**(evals_dB/10)

def normalize_evals(evals):
    return evals/tf.reduce_max(evals)

def normalize_evecs(eigenvecs):
    
    """
    eigenvecs: 
    tensor of dimension: (nb_batch, nb_vector, dim_vector, dim_real_imag=2)
        or of dimension: (nb_vector, dim_vector, dim_real_imag=2)
    """
    
    ndims = eigenvecs.shape.ndims
    
    if ndims == 4: 
    
        nb_batch = eigenvecs.shape[0] # does not necessarily exists
        nb_vectors = eigenvecs.shape[1]
        dim_vectors = eigenvecs.shape[2]
        dim_re_im = eigenvecs.shape[3]


        # square eigenvectors
        eigenvecs_squared = eigenvecs**2

        # sum real and imaginary part of all vectors
        vector_norm = tf.reduce_sum(eigenvecs_squared, axis=-1)

        # for all vectors, sum accross all values in vector
        vector_norm = tf.math.sqrt(tf.reduce_sum(vector_norm,  axis=-1))

        # reshape: (nb_batch, nb_vectors) -> (nb_batch, nb_vectors, dim_vectors)
        vector_norm = tf.repeat(vector_norm[:, : ,tf.newaxis], dim_vectors, axis=-1)
        # reshape: (nb_batch, nb_vectors, dim_vectors) -> (nb_batch, nb_vectors, dim_vectors, dim_re_im)
        vector_norm = tf.repeat(vector_norm[:, :, :, tf.newaxis], dim_re_im, axis=-1)

        # use norm to normalize eigenvectors
        normalized_evecs = tf.divide(eigenvecs, vector_norm) 

        return normalized_evecs
    
    elif ndims == 3:
        
        nb_vectors = eigenvecs.shape[0]
        dim_vectors = eigenvecs.shape[1]
        dim_re_im = eigenvecs.shape[2]


        # square eigenvectors
        eigenvecs_squared = eigenvecs**2

        # sum real and imaginary part of all vectors
        vector_norm = tf.reduce_sum(eigenvecs_squared, axis=-1)

        # for all vectors, sum accross all values in vector
        vector_norm = tf.math.sqrt(tf.reduce_sum(vector_norm,  axis=-1))

        # reshape: (nb_vectors) -> (nb_vectors, dim_vectors)
        vector_norm = tf.repeat(vector_norm[ : ,tf.newaxis], dim_vectors, axis=-1)
        # reshape: (nb_vectors, dim_vectors) -> (nb_vectors, dim_vectors, dim_re_im)
        vector_norm = tf.repeat(vector_norm[:, :, tf.newaxis], dim_re_im, axis=-1)

        # use norm to normalize eigenvectors
        normalized_evecs = tf.divide(eigenvecs, vector_norm) 

        return normalized_evecs
        
    else:
        print("invalid nb of dimension encountered")
        return None

