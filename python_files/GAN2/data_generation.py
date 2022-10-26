import numpy as np
import tensorflow as tf
from input_data import WishartCSM
from acoular import MicGeom, SteeringVector, ImportGrid


def get_sample():    
    # create CSM
    f = 343*16

    wcsm = WishartCSM(
        mics=MicGeom( from_file="tub_vogel64_ap1.xml"),
        loc=(0,0,0.5),
        f=f,
        scale=np.eye(1),
        df=1000,
    )

    csm = wcsm.sample_csm()

    # create noise
    wcsm_noise = WishartCSM(
        mics=MicGeom( from_file="tub_vogel64_ap1.xml"),
        loc=(0,0,0.5),
        f=f,
        scale=np.eye(64)*.1,
        df=1000,
    )

    noise = wcsm_noise.sample_complex_wishart()/wcsm_noise.df
    
    # get "realistic" (i.e. noisy) CSM
    realistic_csm = csm + noise

    # extract eigenvalues (as square shape) and eigenvectors
    evals, evecs = tf.linalg.eigh(realistic_csm)


    evecs_tf = tf.stack([tf.math.real(evecs),tf.math.imag(evecs)],axis=2)
    evals_tf = tf.reshape(tf.math.real(evals), (8,8,1))
    
    # cast from float64 to float32
    evecs_tf = tf.cast(evecs_tf, tf.float32)
    evals_tf = tf.cast(evals_tf, tf.float32)

    return evecs_tf, evals_tf

def get_evecs_batch(batch_size):

    return tf.stack([get_sample()[0] for _ in range(batch_size)], axis=0)
    

def get_evals_batch(batch_size):
    
    return tf.stack([get_sample()[1] for _ in range(batch_size)], axis=0)