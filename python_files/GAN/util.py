import numpy as np
import tensorflow as tf

def rounder(nb, digit_to_keep):
    mult = 10 ** digit_to_keep
    return np.round(nb*mult)/mult


