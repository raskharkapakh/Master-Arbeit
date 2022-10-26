import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

from evals_WGANGP2 import evals_WGANGP
from data_generation import get_sample
from beamforming import beamform, get_csm

NUMFREQ = 1
NUMCHANNELS = 64


def get_augmentented_dataset(n_epoch, size_augmentation, size_original_datasets):
    
    """ steps:
    1. train eigenvalues generator
    2. for each real sample, 
            extract eigenvectors
            generate size_augmentation fake eigenvalues
            merge with real eigenvalues 
    """

    # train evals_WGANGP
    evals_wgangp = evals_WGANGP()
    evals_wgangp.train(n_epoch=n_epoch)

    list_csm = []
    first_time = True 
    for _ in range(size_original_datasets):
        tmp_list_csm = []
        real_evecs, real_evals = get_sample()
        
        real_evecs_real = real_evecs[:, :, 0]
        real_evecs_imag = real_evecs[:, :, 1]

        for _ in range(size_augmentation):
            is_real, tmp_evals = evals_wgangp.generate_evals()
            if is_real:
                # reshape eigenvalues from square form (8,8,1) to vector form (64,1) 
                tmp_evals_vec_numpy = np.sort(np.array(tmp_evals[0, :, :, 0]).flatten())
                tmp_evals_vec = tf.convert_to_tensor(tmp_evals_vec_numpy) 


                csm = get_csm(real_evecs_real, real_evecs_imag, tmp_evals_vec).numpy()
                csm = np.reshape(csm, newshape=(NUMFREQ,NUMCHANNELS,NUMCHANNELS))

                tmp_list_csm.append(csm)

        # first time plot real sample and augmented sample
        if first_time and len(tmp_list_csm) > 0:
            
            # convert square real evals to vector
            real_evals_vec_numpy = np.sort(np.array(real_evals[:, :, 0]).flatten())
            real_evals_vec = tf.convert_to_tensor(real_evals_vec_numpy) 

            real_csm = get_csm(real_evecs_real, real_evecs_imag, real_evals_vec).numpy()
            real_csm = np.reshape(real_csm, newshape=(NUMFREQ,NUMCHANNELS,NUMCHANNELS))
            
            augmented_csm = tmp_list_csm[0]

            beamform(real_csm)
            beamform(augmented_csm)
            """
            fig = plt.figure(figsize=(15,5))

            plt.subplot(1,2,1)
            plt.title("Real CSM")
            beamform(real_csm)
            plt.legend()
            
            plt.subplot(1,2,2)
            plt.title("Augmented CSM")
            beamform(augmented_csm)
            plt.legend()

            fig.suptitle(f'Comparison real and augmented CSM', fontsize=16)
            """
            first_time = False


        # merge list 
        for csm in tmp_list_csm:
            list_csm.append(csm)

    return list_csm
            

