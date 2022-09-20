from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from load_dataset import parser

batch_size = 16
num_channels = 1
csm_size = 64  # cross-spectral matrix
csm_shape = (64, 64, 2)  # MxMx2 (real,imag)
latent_dim = 128

# opem dataset from file
tfile = "/home/gaspard/ETHZ/Master_Arbeit/acoupipe_datasets/training_1-10000_csmtriu_1src_he4.0625-1393.4375Hz_ds1-v001_07-Sep-2022.tfrecord"
#"/home/kujawski/datasets_compute4/training_1-5000000_csmtriu_1src_he4.0625-1393.4375Hz_ds1-v001_01-Nov-2021.tfrecord"
dataset = tf.data.TFRecordDataset(filenames=[tfile])
dataset = dataset.map(parser).shuffle(buffer_size=10).batch(batch_size)


# TODO: understand where architecture come from -> see Gerstoft
# Create the discriminator.
discriminator = keras.Sequential(
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

# TODO: understand where architecture come from -> see Gerstoft
# Create the generator.
generator = keras.Sequential(
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


class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, eigenvecs):
        # Unpack the data.
        # Sample random points in the latent space.
        # This is for the generator.
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
        scaled_eigenvecs = tf.divide(generated_eigenvecs, vector_norm)

        combined_eigenvecs = tf.concat([scaled_eigenvecs, eigenvecs], axis=0)

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
        # of the discriminator)!
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


"""
## Training the Conditional GAN
"""

cond_gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.03),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

cond_gan.fit(dataset, steps_per_epoch=100, epochs=10)

for i in range(100):
    random_latent_vectors = tf.random.normal(shape=(1, latent_dim))
    generated_eigenvecs = generator.predict(random_latent_vectors)

    # scale appropriately
    vector_norm = tf.math.sqrt(
        tf.reduce_sum(tf.reduce_sum(generated_eigenvecs**2, 3), 1)
    )[:, :, tf.newaxis]
    scaled_eigenvecs = tf.divide(generated_eigenvecs, vector_norm)

    predictions = discriminator(scaled_eigenvecs)
    print(predictions[0, 0])
    if predictions[0, 0] > 0:
        print("real csm found")
        break
    elif i == 99:
        print("no real csm found")
