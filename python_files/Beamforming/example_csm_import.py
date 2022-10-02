from pylab import figure, imshow, colorbar, newaxis, diag, array
from os import path
from acoular import __file__ as bpath, MicGeom, RectGrid, SteeringVector,\
 BeamformerBase, L_p, ImportGrid
from spectra import PowerSpectraImport

# set up the parameters
f = 4.0625*343
mg = MicGeom( from_file='tub_vogel64_ap1.xml' )
"""
########## this section is only on how to calculate a synthetic CSM #######################

# source positions and rms values of three sources
loc1=(-0.1,-0.1,0.5) 
loc2=(0.15,0,0.5) 
loc3=(0,0.1,0.5)
rms=array([1,0.7,0.5])

# obtain the transfer function and calculate csm
st2 = SteeringVector(
    grid=ImportGrid(gpos_file=array([loc1,loc2,loc3]).T), 
    mics=mg)
H = st2.transfer(f).T # transfer functions for 8000 Hz
H_h = H.transpose().conjugate() # H hermetian
Q = diag(rms)**2 # matrix containing the source strength 
csm = (H@Q.astype(complex)@H_h)[newaxis] # calculate csm

"""


########################## here we inject the existing CSM by using the PowerSpectraImport class ############## 

# generate test data, in real life this would come from an array measurement
ps_import = PowerSpectraImport(csm=csm.copy(), frequencies=f)
rg = RectGrid( x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z=0.5, \
increment=0.01 )
st = SteeringVector(grid=rg, mics=mg)
bb = BeamformerBase( freq_data=ps_import, steer=st, r_diag=False, cached=False )
pm = bb.synthetic( f, 0 )
Lm = L_p( pm )

# show map
figure()
imshow( Lm.T, origin='lower', vmin=Lm.max()-20, extent=rg.extend(), \
interpolation='bicubic')
colorbar()
