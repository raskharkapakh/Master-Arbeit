import tensorflow as tf
from pylab import figure, imshow, colorbar, newaxis, diag, array
from os import path
from acoular import __file__ as bpath, MicGeom, RectGrid, SteeringVector,\
 BeamformerBase, L_p, ImportGrid
from spectra import PowerSpectraImport


def beamform(csm, helmotz_number=4.0625, measurement=False):
    
    mg = MicGeom(from_file='tub_vogel64_ap1.xml')
    scaling = 1.0
    if measurement:
        mg = MicGeom(from_file='tub_vogel64.xml')
        scaling = 1.5
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
    #figure()
    imshow(Lm.T,
            origin='lower',
            vmin=Lm.max()-20,
            extent=rg.extend(),
            interpolation='bicubic')
    colorbar()

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



