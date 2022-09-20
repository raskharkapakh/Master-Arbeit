from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from load_dataset import parser
import keras.backend as K

# hyperparameter
num_channels = 1
csm_size = 64  # cross-spectral matrix
csm_shape = (64, 64, 2)  # MxMx2 (real,imag)




class GAN(keras.Model):
    
    def __init__(self, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator(latent_dim)
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
    # TODO: understand where architecture come from -> it comes from
    # https://keras.io/examples/generative/conditional_gan/
    # Create the discriminator.
    
    def build_discriminator(self):
        return keras.Sequential(
            [
                keras.layers.InputLayer(csm_shape),
                layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.GlobalMaxPooling2D(),
                layers.Dense(1),
            ],
            name="discriminator",
        )

    # TODO: understand where architecture come from -> it comes from https://keras.io/examples/generative/conditional_gan/
    # Create the generator.
    def build_generator(self, latent_dim):
        return keras.Sequential(
            [
                keras.layers.InputLayer((latent_dim,)),
                layers.Dense(128, activation="relu"),
                layers.LayerNormalization(),
                layers.Dense(512, activation="relu"),
                layers.LayerNormalization(),
                layers.Dense(1024, activation="relu"),
                layers.LayerNormalization(),
                layers.Dense(4096 * 2),
                layers.Reshape((64, 64, 2)),
            ],
            name="generator",
        )


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, eigenvecs):

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


        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_eigenvecs)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator (already updated)!)
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

            predictions = self.discriminator(fake_eigenvecs)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }


    def get_csm(self):
        for i in range(100):
            random_latent_vectors = tf.random.normal(shape=(1, self.latent_dim))
            generated_eigenvecs = self.generator.predict(random_latent_vectors)

            # scale appropriately
            vector_norm = tf.math.sqrt(
                tf.reduce_sum(tf.reduce_sum(generated_eigenvecs**2, 3), 1)
            )[:, :, tf.newaxis]
            scaled_eigenvecs = tf.divide(generated_eigenvecs, vector_norm)

            predictions = self.discriminator(scaled_eigenvecs)
            print(predictions[0, 0])
            

            if predictions[0, 0] > 0:
                print("real csm found")

                # printing image:
                print(predictions.shape)
                return scaled_eigenvecs

                
            elif i == 99:
                print("no real csm found")
                return None
