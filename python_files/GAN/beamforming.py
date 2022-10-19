import tensorflow as tf
from pylab import figure, imshow, colorbar, newaxis, diag, array
from os import path
from acoular import __file__ as bpath, MicGeom, RectGrid, SteeringVector,\
 BeamformerBase, L_p, ImportGrid
from spectra import PowerSpectraImport

def beamform(evecs_real, evecs_imag, evals):
        
    # get the CSM from its eigendecomposition
    csm = get_csm(evecs_real=evecs_real, 
                evecs_imag=evecs_imag,
                evals_vec=evals)
    
    # set up the parameters
    f = 4.0625*343
    mg = MicGeom(from_file='tub_vogel64_ap1.xml')

    # generate test data, in real life this would come from an array measurement
    ps_import = PowerSpectraImport(csm=csm.copy(), frequencies=f)
    rg = RectGrid(x_min=-0.5,
                x_max=0.5,
                y_min=-0.5,
                y_max=0.5,z=0.5,
                increment=0.01)
    st = SteeringVector(grid=rg, mics=mg)
    bb = BeamformerBase(freq_data=ps_import,
                        steer=st,
                        r_diag=False,
                        cached=False)
    pm = bb.synthetic(f, 0)
    Lm = L_p(pm)

    # show map
    figure()
    imshow(Lm.T,
            origin='lower',
            vmin=Lm.max()-20,
            extent=rg.extend(),
            interpolation='bicubic')
    colorbar()



    return None

# return CSM from its eigendecomposition
def get_csm(evecs_r, evecs_i, evals_vec):
    evecs = tf.complex(evecs_r, evecs_i)
    evecs_H = tf.linalg.adjoint(evecs)
    evals = tf.linalg.diag(evals_vec)
    tmp = tf.linalg.matmul(evecs,evals)    
    return tf.linalg.matmul(tmp,evecs_H)

