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

from util import rounder, normalize_eigenvecs

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU, LayerNormalization, GlobalMaxPooling2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.metrics import Mean

from tensorflow.keras.constraints import Constraint

import keras.backend as K





import tensorflow as tf

import matplotlib.pyplot as plt

import sys

import numpy as np


# Disable eager execution in order to use "get_weights" function
#tf.compat.v1.disable_eager_execution()

# hyperparameters
latent_dim = 128
batch_size = 16
num_channels = 1
csm_size = 64  # cross-spectral matrix
csm_shape = (csm_size, csm_size, 2)  # MxMx2 (real,imag)
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


class WGAN():
    
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

        # Build the WGAN
        self.wgan = self.build_wgan(self.generator, self.critic)
     
    
    # Create the critic.
    def build_critic(self):
        
        const = ClipConstraint(0.01)

        model = Sequential()

        model.add(Conv2D(64, (3, 3), strides=(2, 2), input_shape=self.csm_shape, padding="same", kernel_constraint=const))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same", kernel_constraint=const))
        model.add(LeakyReLU(alpha=0.2))
        model.add(GlobalMaxPooling2D())
        model.add(Dense(1, activation='linear'))

        model.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)
	    
        return model


    # Create the generator.
    def build_generator(self):

        model = Sequential()
        model.add(Dense(128, activation="relu", input_dim=self.latent_dim))
        model.add(LayerNormalization())
        model.add(Dense(512, activation="relu"))
        model.add(LayerNormalization())
        model.add(Dense(1024, activation="relu"))
        model.add(LayerNormalization())
        model.add(Dense(4096*2))
        model.add(Reshape((64, 64, 2)))

        return model
 

    def build_wgan(self, generator, critic):
        # make weights in the critic not trainable
        for layer in critic.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(generator)
        # add the critic
        model.add(critic)
        # compile model
        model.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)
        return model



    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, c_optimizer, g_optimizer, loss_fn):
        super(WGAN, self).compile()
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
                    break
                
                c_loss1 = self.critic.train_on_batch(real_sample, real)
                c1_tmp.append(c_loss1)
                
                # generate 'fake' examples

                # create random vector to generate fake sample
                seed = tf.random.normal(shape=(self.batch_size, self.latent_dim))

                # generate fake sample using random seed
                fake_sample = self.generator(seed)

                # normalize eigenvecs to unit length
                norm_fake_sample = normalize_eigenvecs(fake_sample)



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
            g_loss = self.wgan.train_on_batch(seed, real)
            g_hist.append(g_loss)


            # print loss per epoch
            print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (e+1, c1_hist[-1], c2_hist[-1], g_loss))
           
        # line plots of loss
        plt.plot(c1_hist, label='crit_real')
        plt.plot(c2_hist, label='crit_fake')
        plt.plot(g_hist, label='gen')
        plt.legend()
        
    """
    def train(self, n_epoch, data):

        # create batch iterator
        batch = iter(data)

        # groundtruth
        real = -np.ones((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))

        for epoch in range(n_epoch):

            # ======================================================== 
            # train critic (n_critic times per epoch)
            # ========================================================

            for _ in range(self.n_critic):
                
                real_eigenvecs = 0
                
                # try to get next batch of data, if there are None left, break
                try:
                    real_eigenvecs = batch.get_next() #(size batch_size X csm_shape, i.e 16x64x64)
                except:
                    print("There are not sufficient batches to train further. Training interrupted")
                    break

                # create random vector to generate fake eigenvectors
                seed = tf.random.normal(shape=(self.batch_size, self.latent_dim))

                # generate fake eigenvectors using random seed
                generated_eigenvecs = self.generator(seed)

                # normalize eigenvecs to unit length
                vector_norm = tf.math.sqrt(
                    tf.reduce_sum(tf.reduce_sum(generated_eigenvecs**2, 3), 1)
                )
                vector_norm = tf.repeat(vector_norm[:, tf.newaxis, :], 64, axis=1)[
                    :, :, :, tf.newaxis
                ]
                vector_norm = tf.concat([vector_norm, vector_norm], axis=3)  # real, imag
                scaled_generated_eigenvecs = tf.divide(generated_eigenvecs, vector_norm)
                
                # train 
                c_loss_real = self.critic.train_on_batch(real_eigenvecs, real)
                c_loss_fake = self.critic.train_on_batch(scaled_generated_eigenvecs, fake)
                c_loss = 0.5 * np.add(c_loss_fake, c_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)


            # ========================================================
            # train generator
            # ========================================================
            
            # TODO: need to redefine train on batch -> eigenvector need to be normalized
            g_loss  = self.combined.train_on_batch(seed, real)

            # Plot the progress
            print(f"Epoch {epoch}: [C_loss: {rounder(c_loss[0], 3)}] [G_loss: {rounder(g_loss[0],3)}]")

            #print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - c_loss[0], 1 - g_loss[0]))

            # Monitor loss.
            self.gen_loss_tracker.update_state(g_loss)
            self.disc_loss_tracker.update_state(c_loss)
        
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "c_loss": self.disc_loss_tracker.result(),
        }
    """       



    """
    def train_step(self, eigenvecs):

        # ========================================================
        # Train Critic
        # ========================================================

        # for everytime we train the generator, the critic need to be trained n_critic (e.g. 5) times
        for _ in range(self.n_critic):

            # Sample random points in the latent space. This is for the generator to create fake eigenvectors
            batch_size = tf.shape(eigenvecs)[0] 
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            
            # Decode the noise (guided by labels) to fake images.
            generated_eigenvecs = self.generator(random_latent_vectors)

            # normalize eigenvecs to unit length
            vector_norm = tf.math.sqrt(
                tf.reduce_sum(tf.reduce_sum(generated_eigenvecs**2, 3), 1)
            )
            vector_norm = tf.repeat(vector_norm[:, tf.newaxis, :], 64, axis=1)[
                :, :, :, tf.newaxis
            ]
            vector_norm = tf.concat([vector_norm, vector_norm], axis=3)  # real, imag
            scaled_generated_eigenvecs = tf.divide(generated_eigenvecs, vector_norm)

            # Combine the fake (i.e. generated) eigenvectors and real eigenvectors
            combined_eigenvecs = tf.concat([scaled_generated_eigenvecs, eigenvecs], axis=0)

            labels = tf.concat(
                [tf.ones((batch_size, 1)), -tf.ones((batch_size, 1))], axis=0
            )

            # Train the critic.
            with tf.GradientTape() as tape:
                predictions = self.critic(combined_eigenvecs)
                c_loss = self.loss_fn(labels, predictions)
            grads = tape.gradient(c_loss, self.critic.trainable_weights)
            self.c_optimizer.apply_gradients(
                zip(grads, self.critic.trainable_weights)
            )
            
            # the weights of the critic need to be clipped (i.e. kept in a range)
            for l in self.critic.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                l.set_weights(weights)
            


        


        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images".
        # TODO update
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            fake_eigenvecs = self.generator(random_latent_vectors)
            # scale
            vector_norm = tf.math.sqrt(
                tf.reduce_sum(tf.reduce_sum(fake_eigenvecs**2, 3), 1)
            )
            vector_norm = tf.repeat(vector_norm[:, tf.newaxis, :], 64, axis=1)[
                :, :, :, tf.newaxis
            ]
            vector_norm = tf.concat([vector_norm, vector_norm], axis=3)  # real, imag
            fake_eigenvecs = tf.divide(fake_eigenvecs, vector_norm)

            predictions = self.critic(fake_eigenvecs)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(c_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "c_loss": self.disc_loss_tracker.result(),
        }
    """


    def get_sample(self):
        for i in range(100):
            random_latent_vectors = tf.random.normal(shape=(1, self.latent_dim))
            generated_eigenvecs = self.generator.predict(random_latent_vectors)
            scaled_eigenvecs = normalize_eigenvecs(generated_eigenvecs)
            
            predictions = self.critic(scaled_eigenvecs)
            print(predictions[0, 0])
            

            if predictions[0, 0] > 0:
                print("real sample found")

                # printing image:
                print(predictions.shape)
                return scaled_eigenvecs

                
            elif i == 99:
                print("no real sample found")
                return None
