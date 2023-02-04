import tensorflow as tf
import matplotlib.pyplot as plt
from pylab import figure, imshow, colorbar, newaxis, diag, array
from os import path
from acoular import __file__ as bpath, MicGeom, RectGrid, SteeringVector,\
 BeamformerBase, L_p, ImportGrid
from spectra import PowerSpectraImport
import numpy as np


def beamform(csm, helmotz_number=4.0625, measurement=False):
    
    mg = MicGeom(from_file='tub_vogel64_ap1.xml')
    scaling = 1.0
    if measurement:
        mg = MicGeom(from_file='tub_vogel64.xml')
        scaling = 1.4648587220804408#1.5
    f = (helmotz_number*343)/scaling
    # generate test data, in real life this would come from an array measurement
    
    ps_import = PowerSpectraImport(csm=csm.copy(), frequencies=f)
    rg = RectGrid(x_min=-0.5*scaling,
                x_max=0.5*scaling,
                y_min=-0.5*scaling,
                y_max=0.5*scaling,
                z=0.5*scaling,
                increment=0.01)
    st = SteeringVector(grid=rg, mics=mg)
    bb = BeamformerBase(freq_data=ps_import,
                        steer=st,
                        r_diag=False,
                        cached=False)
    pm = bb.synthetic(f, 0)
    Lm = L_p(pm)
    
    # show map
    
    #fig = plt.figure(figsize=(10,10))
    nb_fontsize = 20
    

    im = imshow(Lm.T,
            origin='lower',
            vmin=Lm.max()-20,
            extent=rg.extend(),
            interpolation='bicubic')
    
    plt.xlabel(r"$x$", fontsize=nb_fontsize)
    plt.ylabel(r"$y$", fontsize=nb_fontsize)
    plt.xticks(fontsize=nb_fontsize)
    plt.yticks(fontsize=nb_fontsize)
    
    cbar = colorbar(im)
    im.figure.axes[1].tick_params(labelsize=nb_fontsize)
    cbar.set_label(r'$L_p/$dB', rotation=90, fontsize=nb_fontsize)

    return None

def beamform_difference(csm1, csm2, helmotz_number=4.0625, measurement=False):
    
    mg = MicGeom(from_file='tub_vogel64_ap1.xml')
    scaling = 1.0
    if measurement:
        mg = MicGeom(from_file='tub_vogel64.xml')
        scaling = 1.5
    f = (helmotz_number*343)/scaling
    # generate test data, in real life this would come from an array measurement
    
    ps_import1 = PowerSpectraImport(csm=csm1.copy(), frequencies=f)
    ps_import2 = PowerSpectraImport(csm=csm2.copy(), frequencies=f)
    
    rg = RectGrid(x_min=-0.5*scaling,
                x_max=0.5*scaling,
                y_min=-0.5*scaling,
                y_max=0.5*scaling,
                z=0.5*scaling,
                increment=0.01)
    st = SteeringVector(grid=rg, mics=mg)
    
    bb1 = BeamformerBase(freq_data=ps_import1,
                        steer=st,
                        r_diag=False,
                        cached=False)
    bb2 = BeamformerBase(freq_data=ps_import2,
                        steer=st,
                        r_diag=False,
                        cached=False)
    
    pm1 = bb1.synthetic(f, 0)
    pm2 = bb2.synthetic(f, 0)
    
    Lm1 = L_p(pm1)
    Lm2 = L_p(pm2)

    print(type(Lm1))
    
    diff = Lm1.T - Lm2.T
    
    diff = np.flip(diff, axis=0)
    
    # show difference between the two beamforming maps
    
    nb_fontsize = 20
    
  
    
    im = imshow(diff,
           vmin=diff.min(),
           vmax=diff.max(),
           extent=rg.extend(),
            interpolation='bicubic')
    
    plt.xlabel(r"$x$", fontsize=nb_fontsize)
    plt.ylabel(r"$y$", fontsize=nb_fontsize)
    plt.xticks(fontsize=nb_fontsize)
    plt.yticks(fontsize=nb_fontsize)
    
    cbar = colorbar(im)
    im.figure.axes[1].tick_params(labelsize=nb_fontsize)
    cbar.set_label(r'$L_p/$dB', rotation=90, fontsize=nb_fontsize)

    return None

# return CSM from its eigendecomposition
def get_csm(evecs_real, evecs_imag, evals_vec):
    """
    shape, dytpe:
    - evecs_real: (64, 64), float32
    - evecs_imag: (64, 64), float32
    - evals_vec: (64, ), float32
    """

    evecs = tf.complex(evecs_real, evecs_imag)
    evecs_H = tf.linalg.adjoint(evecs)
    evals_real = tf.linalg.diag(evals_vec)
    evals_imag = tf.zeros(shape=evals_real.shape)
    evals = tf.complex(evals_real, evals_imag)
    tmp = tf.linalg.matmul(evecs,evals)    
    return tf.linalg.matmul(tmp,evecs_H)

def get_rank_I_csm(main_evec_real, main_evec_imag, evals_vec):
    
    main_evec = tf.complex(main_evec_real, main_evec_imag)[:, tf.newaxis]
    main_evec_H = tf.linalg.adjoint(main_evec) # hermitian
        
    evals = tf.complex(evals_vec[-1], 0.0)[tf.newaxis, tf.newaxis]
    
    return main_evec @ evals @ main_evec_H



