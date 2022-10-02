from __future__ import print_function, division



# TensorFlow/Keras import
import tensorflow as tf
import keras.backend as K
#from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU, LayerNormalization, GlobalMaxPooling2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.metrics import Mean
from tensorflow.keras.constraints import Constraint

# Imports
import matplotlib.pyplot as plt
import numpy as np

# My imports
from eigenmanipulation2 import normalize_evecs



# Disable eager execution in order to use "get_weights" function
#tf.compat.v1.disable_eager_execution()

# hyperparameters
latent_dim = 128
batch_size = 16
num_channels = 2
evecs_size = 64  # cross-spectral matrix
evecs_shape = (evecs_size, evecs_size, 2)  # MxMx2 (real,imag)
n_critic = 5
critic_clip_value = 0.1


# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return K.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}


class evecs_WGAN():
    
    def __init__(self):

        # save hyperparameter in model
        self.gen_loss_tracker = Mean(name="generator_loss")
        self.disc_loss_tracker = Mean(name="critic_loss")
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.evecs_size = evecs_size
        self.evecs_shape = evecs_shape

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = n_critic
        self.clip_value = critic_clip_value
        self.optimizer = RMSprop(learning_rate=0.00005,
                            momentum=0.0)

        # Build and compile the critic
        self.critic = self.build_critic()

        # Build the generator
        self.generator = self.build_generator()

        # Build the WGAN
        self.wgan = self.build_wgan(self.generator, self.critic)
     
    
    # Create the critic.
    def build_critic(self):
        
        const = ClipConstraint(0.01)

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.evecs_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same", kernel_constraint=const))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", kernel_constraint=const))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same", kernel_constraint=const))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))

        model.summary()

        sample = Input(shape=self.evecs_shape)
        validity = model(sample)

        model = Model(sample, validity)
        model.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)


        return model


    # Create the generator.
    def build_generator(self):
        
        model = Sequential()
        
        model.add(Dense(128 * 16 * 16, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((16, 16, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(num_channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        sample = model(noise)

        model = Model(noise, sample)
        model.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)


        return model
 

    def build_wgan(self, generator, critic):
        
        z = Input(shape=(self.latent_dim,))
        evecs = generator(z)

        # For the combined model we will only train the generator
        critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = critic(evecs)

        # The combined model  (stacked generator and critic)
        combined = Model(z, valid)
        combined.compile(loss=self.wasserstein_loss,
            optimizer=self.optimizer)
        
        return combined




    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    
    def train(self, evecs_dataset, n_epoch):

        # create batch iterator
        batch = iter(evecs_dataset)

        # groundtruth
        real = -np.ones((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))

        
        # lists for keeping track of loss
        c1_hist, c2_hist, g_hist = list(), list(), list()
        
        for e in range(n_epoch):
            
            c1_tmp, c2_tmp = list(), list()
            

            # ========================================================
            # Train Critic (n_critic times)
            # ========================================================
            
            for _ in range(self.n_critic):
                
                # try to get next batch of data, if there are none left, break
                try:
                    real_evecs = batch.get_next() #(size batch_size X evecs_shape, i.e 16x64x64)
                except:
                    print("There are not sufficient batches to train further. Training interrupted")
                    exit
                
                c_loss1 = self.critic.train_on_batch(real_evecs, real)
                c1_tmp.append(c_loss1)
                
                # generate 'fake' examples                
                seed = tf.random.normal(shape=(self.batch_size, self.latent_dim)) # create random vector to generate fake eigenvectors
                fake_evecs = self.generator(seed) # generate fake sample using random seed
                norm_fake_evecs = normalize_evecs(fake_evecs) # normalize eigenvecs to unit length


                # update critic model weights
                c_loss2 = self.critic.train_on_batch(norm_fake_evecs, fake)
                c2_tmp.append(c_loss2)
            
            # store critic loss
            c1_hist.append(np.mean(c1_tmp))
            c2_hist.append(np.mean(c2_tmp))
            
            # ========================================================
            # Train Generator
            # ========================================================
            
            # prepare points in latent space as input for the generator
            seed = tf.random.normal(shape=(self.batch_size, self.latent_dim))
            # update the generator via the critic's error
            g_loss = self.wgan.train_on_batch(seed, real)
            g_hist.append(g_loss)

            # print loss per epoch
            print('>epoch %d [c_real=%.3f][c_fake=%.3f][g=%.3f]' % (e+1, c1_hist, c2_hist, g_loss))
            print('===========')
           
        # line plots of loss
        
        plt.plot(c1_hist, label='crit_real')
        plt.plot(c2_hist, label='crit_fake')
        plt.plot(g_hist, label='gen')
        plt.legend()
        




    def get_evecs(self):
        for i in range(100):
            random_latent_vectors = tf.random.normal(shape=(1, self.latent_dim))
            generated_evecs = self.generator.predict(random_latent_vectors)
            scaled_generated_evecs = normalize_evecs(generated_evecs) # normalize eigenvecs to unit length
            
            
            predictions = self.critic(scaled_generated_evecs)

            if predictions[0, 0] < 0:
                print("real eigenvalues found")
                return scaled_generated_evecs

            elif i == 99:
                print("no real eigenvalues found")
                return None
