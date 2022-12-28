import numpy as np
import tensorflow as tf
from input_data import WishartCSM
from acoular import MicGeom, SteeringVector, ImportGrid

from eigenmanipulation import normalize_evals, levelify

def get_sample2(loc, helmotz_number):
    # create CSM
    
    f = 343*helmotz_number

    wcsm = WishartCSM(
        mics=MicGeom( from_file="tub_vogel64_ap1.xml"),
        loc=loc,
        f=f,
        scale=np.eye(1),
        df=64*4, # degrees of freedom
    )
    
    
    csm = wcsm.sample_csm()
    
    # create noise
    wcsm_noise = WishartCSM(
        mics=MicGeom( from_file="tub_vogel64_ap1.xml"),
        loc=(0,0,0.5),
        f=f,
        scale=np.eye(64)*.1,
        df=64*4,
    )

    noise = wcsm_noise.sample_complex_wishart()/wcsm_noise.df
    
    # get "realistic" (i.e. noisy) CSM
    realistic_csm = csm + noise


    # extract eigenvalues (as square shape) and eigenvectors
    evals, evecs = tf.linalg.eigh(realistic_csm)

    evecs_real = tf.math.real(evecs)
    evecs_imag = tf.math.imag(evecs)


    # cast from float64 to float32
    evecs_real = tf.cast(evecs_real, tf.float32)
    evecs_imag = tf.cast(evecs_imag, tf.float32)
    evals_tf = tf.cast(evals, tf.float32)
    
    # get evals as level
    evals_dB_tf = tf.convert_to_tensor(levelify(normalize_evals(tf.math.real(evals_tf))))

    # get index of eigenvalues big enough to be considered not a noise representation 
    threshold = 0.1
    index_main = tf.where(normalize_evals(evals_tf) > threshold)
    index_noise = tf.where(normalize_evals(evals_tf) <= threshold)
    
    # separate eigenvecs between main component and noise
    main_evecs_real = tf.gather(evecs_real, index_main)[:,0,:]
    main_evecs_imag = tf.gather(evecs_imag, index_main)[:,0,:]
    noise_evecs_real = tf.gather(evecs_real, index_noise)[:,0,:]
    noise_evecs_imag = tf.gather(evecs_imag, index_noise)[:,0,:]


    main_evecs = tf.stack([main_evecs_real, main_evecs_imag], axis=2)
    noise_evecs = tf.stack([noise_evecs_real, noise_evecs_imag], axis=2)

    # reshape evals into "image" format instead of vetcor format  
    evals_tf = tf.reshape(tf.math.real(evals_tf), (8,8,1))
    evals_dB_tf = tf.reshape(evals_dB_tf, (8,8,1))

    return main_evecs, noise_evecs, evals_tf, evals_dB_tf




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