import tensorflow as tf
import numpy as np
from scipy import linalg
from numpy.random import RandomState
from acoular import SteeringVector, ImportGrid


class WishartCSM:
    def __init__(self,mics,f,scale, df, loc, rng=RandomState(1)):
        self.mics = mics
        self.f = f
        self.scale = scale
        self.df = df
        self.loc = loc
        self.rng = rng
        self.steer = SteeringVector(
            grid=ImportGrid(gpos_file=np.array([loc]).T),mics=mics)                    

    def sample_complex_wishart(self):
        dim = self.scale.shape[0]
        n_tril = dim * (dim-1) // 2
        C = linalg.cholesky(self.scale, lower=True)
        covariances = self.rng.normal(size=n_tril) + 1j*self.rng.normal(size=n_tril)
        # diagonal elements follow random gamma distribution (according to Nagar and Gupta, 2011)
        variances = (np.r_[[self.rng.gamma(self.df-dim +i, scale=1,size=1)**0.5
                        for i in range(dim)]])
        A = np.zeros(C.shape,dtype=complex)
        # input the covariances
        tril_idx = np.tril_indices(dim, k=-1)
        A[tril_idx] = covariances
        # Input the variances
        A[np.diag_indices(dim)] = variances.astype(complex)[:,0]
        # build matrix
        CA = np.dot(C, A)
        return np.dot(CA, CA.conjugate().T)

    def sample_csm(self):
        Q = self.sample_complex_wishart()/self.df
        H = self.steer.transfer(self.f).T # transfer functions for 8000 Hz
        H_h = H.transpose().conjugate() # H hermetian
        return (H@Q.astype(complex)@H_h)

if __name__ == "__main__":
    from acoular import SteeringVector, MicGeom, ImportGrid
    import matplotlib.pyplot as plt

    f = 343*16
    mg = MicGeom( from_file="tub_vogel64_ap1.xml")

    wcsm = WishartCSM(
        mics=mg,
        loc=(0,0,0.5),
        f=f,
        scale=np.eye(1),
        df=100,
    )
    csm = wcsm.sample_csm()

    plt.figure()
    eig, vec = np.linalg.eigh(csm)
    plt.plot(10*np.log10(np.real(eig)[::-1]))
