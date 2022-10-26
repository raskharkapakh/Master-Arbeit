
from __future__ import print_function, division

# Library imports
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

# Imports from TensorFlow/Keras
from tensorflow.keras.layers import Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.metrics import Accuracy

# My imports
from eigenmanipulation2 import normalize_evals
from data_generation import get_evals_batch


num_channels = 1
evals_shape = (8,8,1)
latent_dim = 128
batch_size = 16
n_critic = 5

# learning rate constant
LR = 1e-4
MIN_LR = 1e-6 # Minimum value of learning rate
DECAY_FACTOR=1.00004

class evals_WGANGP():
    def __init__(self):
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.evals_shape = evals_shape
        self.n_critic = n_critic

        """
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.optimizer = RMSprop(lr=0.00005)
        """

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()
        
        self.g_optimizer = Adam(learning_rate=LR, beta_1=0.5)
        self.c_optimizer = Adam(learning_rate=LR, beta_1=0.5)


    def learning_rate_decay(self, current_lr, decay_factor=DECAY_FACTOR):
        # Calculate new learning rate using decay factor
        new_lr = max(current_lr / decay_factor, MIN_LR)
        return new_lr


    def set_learning_rate(self, new_lr):
        #Set new learning rate to optimizers
        
        K.set_value(self.c_optimizer.lr, new_lr)
        K.set_value(self.g_optimizer.lr, new_lr)

    def get_accuracy(self, pred, groundtruth):
        
        # "round" prediction so that accuracy can be computed
        # Because we have a critic instead of a discriminator, accuracy will always be zero otherwise  
        numpy_pred = pred.numpy()
        round_numpy_pred = np.where(numpy_pred < 0, -1, 1)
        round_pred = tf.convert_to_tensor(round_numpy_pred)
        
        
        acc = Accuracy()
        acc.update_state(round_pred, groundtruth)
        accuracy = acc.result().numpy()
        
        return accuracy


    def get_next_batch(self, iterator):
        # get real input data
        try:
            batch = iterator.get_next() #(size batch_size X evals_shape, i.e 16x64)
        except:
            print("There are not sufficient batches to train further. Training interrupted")
            exit()

        return batch, iterator

    
    def build_generator(self):
       
        model = Sequential(
            [
                Dense(128 * 2 * 2, activation="relu", input_dim=self.latent_dim),
                Reshape((2, 2, 128)),
                UpSampling2D(),
                Conv2D(128, kernel_size=4, padding="same"),
                BatchNormalization(momentum=0.8),
                Activation("relu"),
                UpSampling2D(),
                Conv2D(64, kernel_size=4, padding="same"),
                BatchNormalization(momentum=0.8),
                Activation("relu"),
                Conv2D(self.num_channels, kernel_size=4, padding="same"),
                Activation("tanh"),
            ],
            name='generator',
        )

        return model


    def train_generator(self):

        seed = tf.random.normal([self.batch_size, self.latent_dim])
        ###################################
        # Train G
        ###################################
        with tf.GradientTape() as g_tape:
            fake_evals = self.generator([seed], training=True)
            norm_fake_evals = normalize_evals(fake_evals)
            fake_pred = self.critic([norm_fake_evals], training=True)
            g_loss = -tf.reduce_mean(fake_pred)
        # Calculate the gradients for generator
        g_gradients = g_tape.gradient(g_loss,
                                                self.generator.trainable_variables)
        # Apply the gradients to the optimizer
        self.g_optimizer.apply_gradients(zip(g_gradients,
                                                    self.generator.trainable_variables))
        groundtruth_real = -tf.ones(shape=fake_pred.shape)
        g_acc = self.get_accuracy(fake_pred, groundtruth_real)

        return g_loss, g_acc



    def build_critic(self):
        
        model = Sequential(
            [
                Conv2D(16, kernel_size=3, strides=2, input_shape=self.evals_shape, padding="same"),
                LeakyReLU(alpha=0.2),
                Dropout(0.25),
                Conv2D(32, kernel_size=3, strides=2, padding="same"),
                ZeroPadding2D(padding=((0,1),(0,1))),
                BatchNormalization(momentum=0.8),
                LeakyReLU(alpha=0.2),
                Dropout(0.25),
                Conv2D(64, kernel_size=3, strides=2, padding="same"),
                BatchNormalization(momentum=0.8),
                LeakyReLU(alpha=0.2),
                Dropout(0.25),
                Conv2D(128, kernel_size=3, strides=1, padding="same"),
                BatchNormalization(momentum=0.8),
                LeakyReLU(alpha=0.2),
                Dropout(0.25),
                Flatten(),
                Dense(1)
            ],
            name= 'critic',
        )

        return model
    

    def train_critic(self, real_evals):

        lambda_gp = 10 # value advised by paper

        seed = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        alpha = tf.random.uniform(self.evals_shape, minval=0, maxval=1)


        with tf.GradientTape(persistent=True) as d_tape:
            with tf.GradientTape() as gp_tape:
                fake_evals = self.generator([seed], training=True)
                
        
                interpolated_evals = (alpha * real_evals) + ((1 - alpha) * fake_evals)
                norm_interpolated_evals = normalize_evals(interpolated_evals)

                interpolated_evals_pred = self.critic([norm_interpolated_evals], training=True)
            
            norm_real_evals = normalize_evals(real_evals)
            norm_fake_evals = normalize_evals(fake_evals)

            # Compute gradient penalty
            grads = gp_tape.gradient(interpolated_evals_pred, interpolated_evals)
            grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))
            
            fake_pred = self.critic([norm_fake_evals], training=True)
            real_pred = self.critic([norm_real_evals], training=True)
            
            c_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + lambda_gp * gradient_penalty
        # Calculate the gradients for discriminator
        c_gradients = d_tape.gradient(c_loss,
                                                self.critic.trainable_variables)
        # Apply the gradients to the optimizer
        self.c_optimizer.apply_gradients(zip(c_gradients,
                                                    self.critic.trainable_variables))
        
        #groundtruth
        groundtruth_real = -tf.ones(shape=real_pred.shape)
        groundtruth_fake = tf.ones(shape=fake_pred.shape)
        
        c_acc_real = self.get_accuracy(real_pred, groundtruth_real)
        c_acc_fake = self.get_accuracy(real_pred, groundtruth_fake)
        c_acc = np.mean([c_acc_real, c_acc_fake])

        return c_loss, c_acc


    def train(self, n_epoch):        
        
        current_learning_rate = LR
        # create iterator to iterate over batches

        c_loss_list, g_loss_list = [], []
        c_acc_list, g_acc_list = [], [] 

        for e in range(n_epoch):
        
            # ================
            # train critic
            # ================
            
            c_loss_temp_list = []
            c_acc_temp_list = []

            for _ in range(self.n_critic):
                
                # get data
                real_evals = get_evals_batch(batch_size=self.batch_size)

                # Using learning rate decay
                current_learning_rate = self.learning_rate_decay(current_learning_rate)
                self.set_learning_rate(current_learning_rate)

                c_loss_temp, c_acc_temp = self.train_critic(real_evals)
                c_loss_temp_list.append(c_loss_temp)
                c_acc_temp_list.append(c_acc_temp)

            c_loss = np.mean(c_loss_temp_list)
            c_acc = np.mean(c_acc_temp_list)
           

            # ================
            # train generator
            # ================

            # Using learning rate decay
            current_learning_rate = self.learning_rate_decay(current_learning_rate)
            self.set_learning_rate(current_learning_rate)

            g_loss, g_acc = self.train_generator()

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
        plt.legend()

        plt.subplot(1,2,2)
        plt.title("Accuracies")
        plt.plot(np.arange(n_epoch), c_acc_list, label='crit')
        plt.plot(np.arange(n_epoch), g_acc_list, label='gen')
    
        plt.legend()
        

    def generate_evals(self):
        for i in range(100):
            random_latent_vectors = tf.random.normal(shape=(1, self.latent_dim))
            generated_eigenvals = self.generator.predict(random_latent_vectors)
            norm_generated_eigenvals = normalize_evals(generated_eigenvals) # scale appropriately
            predictions = self.critic(norm_generated_eigenvals)
            print(predictions[0, 0])
            
            if predictions[0, 0] > 0:
                print("real eigenvalues found")
                return True, norm_generated_eigenvals
                
            elif i == 99:
                print("no real eigenvalues found")
                return False, norm_generated_eigenvals
