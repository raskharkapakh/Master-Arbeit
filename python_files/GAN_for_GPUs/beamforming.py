import tensorflow as tf
from pylab import figure, imshow, colorbar, newaxis, diag, array
from os import path
from acoular import __file__ as bpath, MicGeom, RectGrid, SteeringVector,\
 BeamformerBase, L_p, ImportGrid
from spectra import PowerSpectraImport

def beamform(csm):
        
    # set up the parameters
    f = 4.0625*343 
    # TODO: f should be modified here to beonly on
    mg = MicGeom(from_file='tub_vogel64_ap1.xml')


    """
    TODO: CSM shape must be (numfrequencies, numchannels, numchannels)
    TODO: here I changed "csm.copy()" to only "csm" -> see if it matters
    TODO: check what type "csm" should be.
    """
    
    NUMFREQ = 1 # one freq with index bin = 13 ? -> is that accurate ?
    NUMCHANNELS = 64
    

    csm = tf.reshape(csm, shape=(NUMFREQ,NUMCHANNELS,NUMCHANNELS))

    # generate test data, in real life this would come from an array measurement
    
    ps_import = PowerSpectraImport(csm=csm, frequencies=f)
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
def get_csm(evecs_real, evecs_imag, evals_vec):
    evecs = tf.complex(evecs_real, evecs_imag)
    evecs_H = tf.linalg.adjoint(evecs)
    evals_real = tf.linalg.diag(evals_vec)
    evals_imag = tf.zeros(shape=evals_real.shape)
    evals = tf.complex(evals_real, evals_imag)
    tmp = tf.linalg.matmul(evecs,evals)    
    return tf.linalg.matmul(tmp,evecs_H)

