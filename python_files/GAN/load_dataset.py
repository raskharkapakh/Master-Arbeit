import tensorflow as tf
import os

def uncompress_csm(csm_compressed):
    csm_compressed = csm_compressed[:,:,0]
    low_diag = tf.linalg.band_part(csm_compressed,num_lower=-1,num_upper=0) - tf.linalg.band_part(csm_compressed,num_lower=0,num_upper=0) # lower diagonal of csm_compressedtriu
    up_diag = tf.linalg.band_part(csm_compressed,num_lower=0,num_upper=-1) - tf.linalg.band_part(csm_compressed,num_lower=0,num_upper=0) # upper diagonal of csm triu
    csm_real = up_diag + tf.transpose(tf.linalg.band_part(csm_compressed,num_lower=0,num_upper=-1),[1,0]) # add the real lower triangular part
    csm_imag = -1*low_diag + tf.transpose(low_diag,[1,0])
    return tf.complex(csm_real,csm_imag)

def get_csm_eigenmodes(csm):
    evls, evecs = tf.linalg.eigh(csm)
    return evecs*evls[tf.newaxis,:] # multiply columns with corresponding eigenvalue

def parser(record):
    parsed = tf.io.parse_single_example(
    record,
    {
    #'loc' : tf.io.FixedLenFeature((2,),tf.float32),
    'csmtriu' : tf.io.FixedLenFeature((64,64,1),tf.float32)}        
    )
    csm = uncompress_csm(parsed['csmtriu'])
    csm /= csm[63,63] # normalize on reference microphone autopower
    evls, evecs = tf.linalg.eigh(csm)
    return tf.stack([tf.math.real(evecs),tf.math.imag(evecs)],axis=2)

def full_parser(record):
    parsed = tf.io.parse_single_example(
    record,
    {
    #'loc' : tf.io.FixedLenFeature((2,),tf.float32),
    'csmtriu' : tf.io.FixedLenFeature((64,64,1),tf.float32)}        
    )
    csm = uncompress_csm(parsed['csmtriu'])
    csm /= csm[63,63] # normalize on reference microphone autopower
    evals, evecs = tf.linalg.eigh(csm)


    # in order to get to have similar dimensions as the eigenvectors, the eigenvalues are stored in a diagonal matrix.
    return tf.stack([tf.math.real(evecs),
                    tf.math.imag(evecs), 
                    tf.linalg.diag(tf.math.real(evals)), 
                    tf.linalg.diag(tf.math.imag(evals))],
                    axis=2)





if __name__ == "__main__":

    batch_size = 1
    tfile = "/home/kujawski/datasets_compute4/training_1-5000000_csmtriu_1src_he4.0625-1393.4375Hz_ds1-v001_01-Nov-2021.tfrecord"

    dataset = tf.data.TFRecordDataset(
            filenames=[tfile])  
    dataset = dataset.map(parser).shuffle(buffer_size=10).batch(batch_size)
    data = next(iter(dataset))

    eigenvecs = data[0,:,:,:]
    vector_norm = np.sqrt((eigenvecs**2).numpy().sum(2).sum(0))
    eigenvecs /= vector_norm[:,tf.newaxis]
