import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

from evals_WGANGP import evals_WGANGP
from data_generation import get_sample
from beamforming import beamform, get_csm

def get_augmentented_dataset(evals_dataset, batch_size, nb_batch, steps_per_epoch, evecs_dataset, size_augmentation):
    
    """ steps:
    1. train eigenvalues generator
    2. for each real sample, 
            extract eigenvectors
            generate size_augmentation fake eigenvalues
            merge with real eigenvalues 
    """
    # create and train WGAN-GP for eigenvalues 
    evals_wgangp = evals_WGANGP()
    evals_wgangp.compile()
    evals_wgangp.fit(
        evals_dataset,
        batch_size=batch_size,
        epochs=nb_batch,
        steps_per_epoch=steps_per_epoch
    )


    list_sample = []

    for evecs in evecs_dataset:
  
        tmp_list_sample = []
  

        for _ in range(size_augmentation):
            is_real, gen_evals = evals_wgangp.generate_evals()

            if is_real:

                evecs_tf = tf.cast(evecs, tf.float32)
                evals_tf = tf.cast(gen_evals, tf.float32)

                sample = (evecs_tf, evals_tf)
                tmp_list_sample.append(sample)
              
        # merge list 
        for s in tmp_list_sample:
            list_sample.append(s)

    return list_sample
            

