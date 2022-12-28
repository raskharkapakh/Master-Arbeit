
from __future__ import print_function, division

# Library imports
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

# Imports from TensorFlow/Keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# My imports
from eigenmanipulation import normalize_evecs



# parameters
num_channels = 1
#evecs_SHAPE = (64,64,2)
evecs_SHAPE = (1,64,2)

latent_dim = 128
batch_size = 16
critic_extra_steps=3
gp_weight=10.0

generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
critic_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)


def critic_loss(real_evecs, fake_evecs):
    real_loss = tf.reduce_mean(real_evecs)
    fake_loss = tf.reduce_mean(fake_evecs)
    return fake_loss - real_loss

def generator_loss(fake_evecs):
    return -tf.reduce_mean(fake_evecs)
"""
# learning rate constant
LR = 1e-4
MIN_LR = 1e-6 # Minimum value of learning rate
DECAY_FACTOR=1.00004
"""


def get_generator_model():
    
    noise = Input(shape=(latent_dim,))
    
    x = Dense(256, use_bias=False)(noise)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    #x = BatchNormalization(momentum=0.8)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    #x = BatchNormalization(momentum=0.8)(x)
    x = Dense(np.prod(evecs_SHAPE), activation='tanh')(x)
    x = Reshape(evecs_SHAPE)(x)
   
    g_model = Model(noise, x, name="generator")

    #g_model.summary()

    return g_model

def get_critic_model():

    evecs = Input(shape=evecs_SHAPE)
    
    x = Flatten(input_shape=evecs_SHAPE)(evecs)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1)(x)
    
    c_model = Model(evecs, x, name="critic")

    #c_model.summary()
    
    return c_model



class main_evecs_WGANGP(Model):
    
    def __init__(self):
        super(main_evecs_WGANGP, self).__init__()
        
        self.critic = get_critic_model()
        self.generator = get_generator_model()
        self.latent_dim = latent_dim
        self.c_steps = critic_extra_steps
        self.gp_weight = gp_weight

    def compile(self):
        super(main_evecs_WGANGP, self).compile()
        self.c_optimizer = critic_optimizer
        self.g_optimizer = generator_optimizer
        self.c_loss_fn = critic_loss
        self.g_loss_fn = generator_loss

    def gradient_penalty(self, batch_size, real_evecs, fake_evecs):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the critic loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_evecs - real_evecs
        interpolated = real_evecs + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the critic output for this interpolated image.
            pred = self.critic(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_evecs):
        if isinstance(real_evecs, tuple):
            real_evecs = real_evecs[0]

        # Get the batch size
        batch_size = tf.shape(real_evecs)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the critic and get the critic loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the critic loss
        # 6. Return the generator and critic losses as a loss dictionary

        # Train the critic first. The original paper recommends training
        # the critic for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.c_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake eigenvectors from the latent vector
                fake_evecs = self.generator(random_latent_vectors, training=True)
                # Normalize the fake eigenvectors
                norm_fake_evecs = normalize_evecs(fake_evecs)
                # Get the logits for the fake eigenvectors
                fake_logits = self.critic(norm_fake_evecs, training=True)
                # Normalize the real eigenvectors
                norm_real_evecs = normalize_evecs(real_evecs)
                # Get the logits for the real eigenvectors
                real_logits = self.critic(norm_real_evecs, training=True)

                # Calculate the critic loss using the fake and real eigenvectors logits
                c_cost = self.c_loss_fn(real_evecs=real_logits, fake_evecs=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, norm_real_evecs, norm_fake_evecs)
                # Add the gradient penalty to the original critic loss
                c_loss = c_cost + gp * self.gp_weight

            # Get the gradients w.r.t the critic loss
            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            # Update the weights of the critic using the critic optimizer
            self.c_optimizer.apply_gradients(
                zip(c_gradient, self.critic.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_evecs = self.generator(random_latent_vectors, training=True)
            # Normalize generated eigenvectors
            norm_generated_evecs = normalize_evecs(generated_evecs)   
            # Get the critic logits for fake images
            gen_evecs_logits = self.critic(norm_generated_evecs, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_evecs_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"c_loss": c_loss, "g_loss": g_loss}

    def generate_evecs(self, nb_trial=100):
        
        for i in range(nb_trial):
            random_latent_vectors = tf.random.normal(shape=(1, self.latent_dim))
            generated_evecs = self.generator.predict(random_latent_vectors)
            generated_evecs = tf.convert_to_tensor(generated_evecs)
            norm_generated_evecs = normalize_evecs(generated_evecs) # scale appropriately
            predictions = self.critic(norm_generated_evecs)
            print(predictions[0, 0])
            
            if predictions[0, 0] > 0:
                print("real main eigenvectors found")
                return True, norm_generated_evecs
                
        
        print("no real main eigenvectors found")
        return False, norm_generated_evecs

