import acoular
import numpy as np
import tensorflow as tf
from eigenmanipulation import normalize_evals, levelify

def get_measurement_sample(measurement_name, freq_index):

    nb_samples = acoular.TimeSamples(name=measurement_name).numsamples
        
    # for data augmentation purposes create the CSM out of different slices of the time serie (or/and by varying parameters such as the block size, window, and overlap)

    # Get a random slice of measurement 
    ts_slice = acoular.MaskedTimeSamples(name=measurement_name)

    nb_slices = 100 # maximum number of non-overlapping slices wished
    slice_length = int(np.floor(nb_samples/nb_slices)) # length of slice to use
    start_index = np.random.randint(0, (nb_samples-slice_length)-1) 
    stop_index = start_index + slice_length

    ts_slice.start = start_index
    ts_slice.stop = stop_index

    # Creating the power spectra (object containing the cross spectral matrix)
    ps = acoular.PowerSpectra(time_data=ts_slice, block_size=128, window='Hanning', overlap="75%")
    #TODO: uncomment if necessary: print(f"measurement #blocks: {ps.num_blocks}")
    # get csm
    csm = ps.csm
    
    

    #freq_index = 64
    freq_csm = csm[freq_index, :, :]

    # extract eigenvalues and eigenvectors from CSM
    main_evecs, noise_evecs, evals, evals_dB_tf = get_evals_evecs(freq_csm)
    
    return main_evecs, noise_evecs, evals, evals_dB_tf

def get_evals_evecs(csm):
    
    evals, evecs = tf.linalg.eigh(csm)

    # =====================================================================================
    evecs_real = tf.math.real(evecs)
    evecs_imag = tf.math.imag(evecs)


    # cast from float64 to float32
    evecs_real = tf.cast(evecs_real, tf.float32)
    evecs_imag = tf.cast(evecs_imag, tf.float32)
    evals_tf = tf.cast(evals, tf.float32)
    
    # get evals as level
    evals_dB_tf = tf.convert_to_tensor(levelify(normalize_evals(tf.math.real(evals_tf))))

    # get index of eigenvalues big enough to be considered not a noise representation 
    #threshold_max = np.max(np.array(evals_tf)) # SEE IF IT APPLIES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    index_main = tf.where(normalize_evals(evals_tf) == 1) # 1 is not a magic number, it is max value after normalization (by def.)
    index_noise = tf.where(normalize_evals(evals_tf) < 1)
    
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