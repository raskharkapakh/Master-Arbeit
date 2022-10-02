"""
from tensorflow import keras
from tensorflow.keras import layers
#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
#from load_dataset import parser
import keras.backend as K
from keras.optimizers import RMSprop
"""
# ===========
from __future__ import print_function, division
from typing import Concatenate

from util import rounder, normalize_eigenvecs

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU, LayerNormalization, GlobalMaxPooling2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.metrics import Mean

from tensorflow.keras.constraints import Constraint

import keras.backend as K

from functools import partial





import tensorflow as tf

import matplotlib.pyplot as plt

import sys

import numpy as np

tf.compat.v1.disable_eager_execution()
# Disable eager execution in order to use "get_weights" function
#tf.compat.v1.disable_eager_execution()

# hyperparameters
latent_dim = 128
batch_size = 16
num_channels = 4
csm_size = 64  # cross-spectral matrix
csm_shape = (csm_size, csm_size, num_channels)  # MxMx2 (real,imag)
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

class RandomWeightedAverage(Add):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        #alpha = K.random_uniform((32, 1, 1, 1))
        shape = inputs[0].shape
        alpha = K.random_uniform(shape)
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGAN_GP():
    
    def __init__(self):

        # save hyperparameter in model
        self.gen_loss_tracker = Mean(name="generator_loss")
        self.disc_loss_tracker = Mean(name="critic_loss")
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.csm_size = csm_size
        self.csm_shape = csm_shape

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = n_critic
        self.clip_value = critic_clip_value
        self.optimizer = RMSprop(learning_rate=0.00005,
                            momentum=0.0)

        # Build and compile the critic
        self.critic = self.build_critic()

        # Build the generator
        self.generator = self.build_generator()

        # Build the WGAN_GP
        #self.wgan_gp = self.build_wgan_gp(self.generator, self.critic)

        
        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_sample = Input(shape=self.csm_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_sample = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_sample)
        valid = self.critic(real_sample)

        # Construct weighted average between real and fake images
        print(f"real_sample shape {real_sample.shape}")
        print(f"fake_sample shape {fake_sample.shape}")

        interpolated_sample = RandomWeightedAverage()([real_sample, fake_sample])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_sample)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_sample)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic = Model(inputs=[real_sample, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=self.optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        sample = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(sample)
        # Defines generator model
        self.generator = Model(z_gen, valid)
        self.generator.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)
        
     
    
    # Create the critic.
    def build_critic(self):
        
        const = ClipConstraint(0.01)

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.csm_shape, padding="same"))
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

        sample = Input(shape=self.csm_shape)
        validity = model(sample)

        final_model = Model(sample, validity)
        #final_model.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)


        return final_model



    # Create the generator.
    def build_generator(self):
        
        model = Sequential()
        
        # here changed a from 7 to 16
        a = 16 
        model.add(Dense(128 * a * a, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((a, a, 128)))
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
        
        #model.add(Reshape((64, 64, 2)))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        sample = model(noise)


        final_model = Model(noise, sample)
        #final_model.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)


        return final_model
 
    """
    def build_wgan_gp(self, generator, critic):
        
        z = Input(shape=(self.latent_dim,))
        img = generator(z)

        # For the combined model we will only train the generator
        critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = critic(img)

        # The combined model  (stacked generator and critic)
        combined = Model(z, valid)
        combined.compile(loss=self.wasserstein_loss,
            optimizer=self.optimizer,
            metrics=['accuracy'])
        
        return combined
    """


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, c_optimizer, g_optimizer, loss_fn):
        super(WGAN_GP, self).compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
    
    def train(self, n_epochs, dataset):

        # create batch iterator
        batch = iter(dataset)

        # groundtruth
        real = -np.ones((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))

        
        # lists for keeping track of loss
        c1_hist, c2_hist, g_hist = list(), list(), list()
        
        for e in range(n_epochs):
            
            c1_tmp, c2_tmp = list(), list()
            

            # ========================================================
            # Train Critic (n_critic times)
            # ========================================================
            
            for _ in range(self.n_critic):
                
                #real_sample = 0 TODO: figure out if necessary
                
                # try to get next batch of data, if there are None left, break
                try:
                    real_sample = batch.get_next() #(size batch_size X csm_shape, i.e 16x64x64)
                except:
                    print("There are not sufficient batches to train further. Training interrupted")
                    exit
                
                c_loss1 = self.critic.train_on_batch(real_sample, real)
                c1_tmp.append(c_loss1)
                
                # generate 'fake' examples

                # create random vector to generate fake sample
                seed = tf.random.normal(shape=(self.batch_size, self.latent_dim))

                # generate fake sample using random seed
                fake_sample = self.generator(seed)

                # split eigenvectors and eigenvalues
                fake_evecs = fake_sample[:,:,:,0:2]
                fake_evals = fake_sample[:,:,:,2:4]

                # normalize eigenvecs to unit length
                norm_fake_evecs = normalize_eigenvecs(fake_evecs)

                # merge again

                
                norm_fake_sample = tf.concat([norm_fake_evecs,fake_evals],axis=3)

                # update critic model weights
                c_loss2 = self.critic.train_on_batch(norm_fake_sample, fake)
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
            g_loss = self.wgan_gp.train_on_batch(seed, real)
            g_hist.append(g_loss[0])

            # print loss per epoch
            print('>epoch %d [c_real=%.3f][c_fake=%.3f][g=%.3f]' % (e+1, c1_hist[-1], c2_hist[-1], g_loss[0]))
            print('===========')
           
        # line plots of loss
        
        plt.plot(c1_hist, label='crit_real')
        plt.plot(c2_hist, label='crit_fake')
        plt.plot(g_hist, label='gen')
        plt.legend()
        
 


    def get_sample(self):
        for i in range(100):
            random_latent_vectors = tf.random.normal(shape=(1, self.latent_dim))
            generated_sample = self.generator.predict(random_latent_vectors)

            # split eigenvectors and eigenvalues
            generated_evecs = generated_sample[:,:,:,0:2]
            generated_evals = generated_sample[:,:,:,2:4]

            # normalize eigenvecs to unit length
            scaled_generated_evecs = normalize_eigenvecs(generated_evecs)

            # merge again
            scaled_sample = tf.concat([scaled_generated_evecs,generated_evals],axis=3)
            
            
            predictions = self.critic(scaled_sample)
            print(predictions[0, 0])
            

            if predictions[0, 0] > 0:
                print("real sample found")

                # printing image:
                print(predictions.shape)
                return scaled_sample

                
            elif i == 99:
                print("no real sample found")
                return None
