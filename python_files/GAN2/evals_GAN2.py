# Import
import matplotlib.pyplot as plt
import numpy as np

# TensorFlow/Keras import
import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Conv2D, UpSampling2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU, LayerNormalization, GlobalMaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

# My import
from eigenmanipulation2 import normalize_evals

# hyperparameter
num_channels = 1
evals_size = (64,)
latent_dim = 128
batch_size = 16

class evals_GAN():
    
    def __init__(self):

        self.evals_size = evals_size
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        self.optimizer = Adam(0.0002, 0.5)

        #self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        #self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")


        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy'])


        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        eval = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(eval)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy',
                                optimizer=self.optimizer,
                                metrics=['accuracy'])
    
    def build_discriminator(self):
        
        model = Sequential()
        
        # new: conv struct
        """
        #model.add(Dense(64, input_shape=self.evals_size))
        model.add(Reshape((8,8,1), input_shape=self.evals_size))
        model.add(Conv2D(16, kernel_size=3, strides=2, padding="same"))
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
        model.add(Dense(1, activation='linear'))
        """
        # old: conv struct
        model.add(Dense(512, input_shape=self.evals_size))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        

        eval = Input(shape=self.evals_size)
        validity = model(eval)

        return Model(eval, validity)

    def build_generator(self):
        
        model = Sequential()
        # new: conv struct
        """
        a = 2 
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
        model.add(Reshape(self.evals_size, input_shape=(8,8,1)))
        """
    

        # TODO: make sure out is of dimension [64]


        # old: conv struct
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.evals_size), activation='tanh'))
        #model.add(Reshape(self.evals_size))
        
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        evals = model(noise)

        model = Model(noise, evals)

        return model

    """
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    """

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]
    """
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
    """
    
    def train(self, evals_dataset, n_epoch):

        # get data
        batch = iter(evals_dataset)

        # groundtruth
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        c1_loss_list, c2_loss_list, g_loss_list = list(), list(), list() 
        c_acc_list, g_acc_list = list(), list()

        for e in range(n_epoch):
            
            # -------------------    
            # train discriminator
            # -------------------

            # train discriminator with real samples
            try:
                real_evals = batch.get_next() #(size batch_size X evals_size, i.e 16x64)
            except:
                print("There are not sufficient batches to train further. Training interrupted")
                exit()


            norm_real_evals = normalize_evals(real_evals)
            c1_loss, c1_acc = self.discriminator.train_on_batch(norm_real_evals, real)
            
            
            
            # train discriminator with fake samples
            
            seed = tf.random.normal(shape=(self.batch_size, self.latent_dim)) # random vector to generate fake sample
            fake_evals = self.generator(seed) # generate fake sample using random seed
            norm_fake_evals = normalize_evals(fake_evals)
            c2_loss, c2_acc = self.discriminator.train_on_batch(norm_fake_evals, fake)
            
            # save loss and accuracy for plotting
            c1_loss_list.append(c1_loss)
            c2_loss_list.append(c2_loss)
            c_acc = np.mean([c1_acc, c2_acc])
            c_acc_list.append(c_acc)
            

            # ---------------    
            # train generator
            # ---------------

            seed = np.random.normal(0, 1, (batch_size, self.latent_dim))

            g_loss, g_acc = self.combined.train_on_batch(seed, real)
            
            # save loss and accuracy for plotting
            g_loss_list.append(g_loss)
            g_acc_list.append(g_acc)
            
            # print status
            print('-=-=- EPOCH %d -=-=-' % (e+1))
            print('>loss: [c_real=%.3f][c_fake=%.3f][g=%.3f]' % (c1_loss, c2_loss, g_loss))
            print('>accuracy: [c=%.3f][g=%.3f]' % (c_acc, g_acc))
            print("")
        
        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)
        plt.title("Losses")
        plt.plot(np.arange(n_epoch), c1_loss_list, label='crit_real')
        plt.plot(np.arange(n_epoch), c2_loss_list, label='crit_fake')
        plt.plot(np.arange(n_epoch), g_loss_list, label='gen')
        plt.legend()
        
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
