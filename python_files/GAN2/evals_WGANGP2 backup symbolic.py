
from __future__ import print_function, division

#from keras.layers.merge import _Merge
#from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D, Layer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from functools import partial

import tensorflow as tf
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

#import sys

import numpy as np

"""
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
"""

# My import
from eigenmanipulation2 import normalize_evals


num_channels = 1
evals_shape = (8,8,1)
latent_dim = 128
batch_size = 16

"""
TODO: build custom layer:

class RandomWeightedAverage(Layer):
    def __init__(self, batch_size, evals_shape):
        super().__init__()
        self.batch_size = batch_size
        self.evals_shape = evals_shape

    def build():
        TODO

    def call(self, inputs, **kwargs):
        shape = (self.batch_size, 
                self.evals_shape[0],
                self.evals_shape[1], 
                self.evals_shape[2])
        alpha = tf.random.uniform(shape)
        #alpha = tf.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]
"""

class RandomWeightedAverage(Layer):
    def __init__(self, batch_size, evals_shape, **kwargs):
        self.shape = (batch_size, 
                    evals_shape[0],
                    evals_shape[1], 
                    evals_shape[2])
        super().__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        super().build(input_shape, **kwargs)

    def call(self, inputs, **kwargs):
        alpha = tf.random.uniform(self.shape)
        #alpha = tf.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, **kwargs):
        return self.shape

"""
class RandomWeightedAverage(_Merge):
    #Provides a (random) weighted average between real and generated image samples
    def _merge_function(self, inputs):
        shape = inputs[0].shape
        alpha = K.random_uniform(shape)
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
"""

class evals_WGANGP():
    def __init__(self):
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.evals_shape = evals_shape

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_evals = Input(shape=self.evals_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_evals = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_evals)
        real = self.critic(real_evals)

        # Construct weighted average between real and fake images
        interpolated_evals = RandomWeightedAverage(self.batch_size, self.evals_shape)([real_evals, fake_evals])
        """
        alpha = K.random_uniform(evals_shape)
        interpolated_evals = (alpha * real_evals) + ((1 - alpha) * fake_evals)
        """
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_evals)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_evals)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_evals, z_disc],
                            outputs=[real, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                optimizer=self.optimizer,
                                loss_weights=[1, 1, 10],
                                metrics=['accuracy'])
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
        evals = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(evals)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=self.optimizer,
        metrics=['accuracy'])


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

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 2 * 2, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((2, 2, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.num_channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))
        """
        model.add(Reshape(self.evals_shape, input_shape=self.evals_img_shape))
        """

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()
        """
        model.add(Reshape(self.evals_img_shape, input_shape=self.evals_shape))"""
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.evals_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.evals_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, evals_dataset, n_epoch):

        # create iterator to iterate over batches
        batch = iter(evals_dataset)

        # groundtruth
        real = -tf.ones((self.batch_size, 1))
        fake = tf.ones((self.batch_size, 1))
        dummy = tf.zeros((batch_size, 1)) # Dummy gt for gradient penalty

        # list for storing performances while training
        c_loss_list, g_loss_list = list(), list() 
        c_acc_list, g_acc_list = list(), list()

        for e in range(n_epoch):

            for _ in range(self.n_critic):
                
                c_loss_list_tmp, c_acc_list_tmp = list(), list()

                try:
                    real_evals = batch.get_next() #(size batch_size X evals_shape, i.e 16x64)
                except:
                    print("There are not sufficient batches to train further. Training interrupted")
                    exit()
                # ---------------------
                #  Train Discriminator
                # ---------------------


                # BEGIN VERSION 1:
                # vvvvvvvvvvvvvvvv
                """
                norm_real_evals = normalize_evals(real_evals)
                c1_loss, c1_acc = self.critic_model.train_on_batch([norm_real_evals], [real, dummy])
                
                # train discriminator with fake samples
                seed = tf.random.normal(shape=(self.batch_size, self.latent_dim)) # random vector to generate fake sample
                fake_evals = self.generator(seed) # generate fake sample using random seed
                norm_fake_evals = normalize_evals(fake_evals)
                c2_loss, c2_acc = self.critic_model.train_on_batch([norm_fake_evals], [fake, dummy])

                c_loss = np.mean([c1_loss, c2_loss])
                c_acc = np.mean([c1_acc, c2_acc])
                """
                # ^^^^^^^^^^^^^
                # END VERSION 1

                # BEGIN VERSION 2:
                # vvvvvvvvvvvvvvvv
                
                # Sample generator input
                noise = tf.random.normal(shape=(batch_size, self.latent_dim), mean=0.0, stddev=1.0)
                # Train the critic
                # TODO: here need to scale fake and real evals
                # c_loss, c_acc
                test = self.critic_model.train_on_batch(
                    [real_evals, noise],
                    [real, fake, dummy]
                )
                
                # ^^^^^^^^^^^^^
                # END VERSION 2

                print(test.shape)
                
                c_loss_list_tmp.append(c_loss)
                c_acc_list_tmp.append(c_acc)

            c_loss = np.mean(c_loss_list_tmp)
            c_acc = np.mean(c_acc_list_tmp)



            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss, g_acc = self.generator_model.train_on_batch(noise, real)

            # save the progress
            c_loss_list.append(c_loss)
            c_acc_list.append(c_acc)
            g_loss_list.append(g_loss)
            g_acc_list.append(g_acc)

            # print the progress
            print('-=-=- EPOCH %d -=-=-' % (e+1))
            print('>loss: [c=%.3f][g=%.3f]' % (c_loss, g_loss))
            print('>accuracy: [c=%.3f][g=%.3f]' % (c_acc, g_acc))
            print("")
        
        # plot the progress
        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)
        plt.title("Losses")
        plt.plot(np.arange(n_epoch), c_loss_list, label='crit')
        plt.plot(np.arange(n_epoch), g_loss_list, label='gen')
        
        plt.subplot(1,2,2)
        plt.title("Accuracies")
        plt.plot(np.arange(n_epoch), c_acc_list, label='crit')
        plt.plot(np.arange(n_epoch), g_acc_list, label='gen')
        plt.legend()

    def get_evals(self):
        for i in range(100):
            random_latent_vectors = tf.random.normal(shape=(1, self.latent_dim))
            generated_eigenvals = self.generator.predict(random_latent_vectors)
            norm_generated_eigenvals = normalize_evals(generated_eigenvals) # scale appropriately
            predictions = self.discriminator(norm_generated_eigenvals)
            print(predictions[0, 0])
            
            if predictions[0, 0] > 0:
                print("real eigenvalues found")
                return True, norm_generated_eigenvals
                
            elif i == 99:
                print("no real eigenvalues found")
                return False, norm_generated_eigenvals
