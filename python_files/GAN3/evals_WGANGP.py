
from __future__ import print_function, division

# Library imports
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

# Imports from TensorFlow/Keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D, Cropping2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import Adam

# My imports
from eigenmanipulation import normalize_evals



# parameters
num_channels = 1
EVALS_SHAPE = (8,8,1)
latent_dim = 128
batch_size = 16
n_critic = 5
critic_extra_steps=3
gp_weight=10.0

generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
critic_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)


def critic_loss(real_evals, fake_evals):
    real_loss = tf.reduce_mean(real_evals)
    fake_loss = tf.reduce_mean(fake_evals)
    return fake_loss - real_loss

def generator_loss(fake_evals):
    return -tf.reduce_mean(fake_evals)

# learning rate constant
LR = 1e-4
MIN_LR = 1e-6 # Minimum value of learning rate
DECAY_FACTOR=1.00004


def conv_block(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    drop_value=0.5,
):
    x = Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = Dropout(drop_value)(x)
    return x


def get_critic_model():
    img_input = Input(shape=EVALS_SHAPE)
    # Zero pad the input to make the input images size to (32, 32, 1).
    x = ZeroPadding2D((2, 2))(img_input)
    x = conv_block(
        x,
        64,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        use_bias=True,
        activation=LeakyReLU(0.2),
        use_dropout=False,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        128,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=LeakyReLU(0.2),
        use_bias=True,
        use_dropout=True,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        256,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=LeakyReLU(0.2),
        use_bias=True,
        use_dropout=True,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        512,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=LeakyReLU(0.2),
        use_bias=True,
        use_dropout=False,
        drop_value=0.3,
    )

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(1)(x)

    c_model = Model(img_input, x, name="critic")
    return c_model



def upsample_block(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(1, 1),
    up_size=(2, 2),
    padding="same",
    use_bn=False,
    use_bias=True,
    use_dropout=False,
    drop_value=0.3,
):
    x = UpSampling2D(up_size)(x)
    x = Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)

    if use_bn:
        x = BatchNormalization()(x)

    if activation:
        x = activation(x)
    if use_dropout:
        x = Dropout(drop_value)(x)
    return x


def get_generator_model():
    noise = Input(shape=(latent_dim,))
    
    a = 64
    x = Dense(4 * 4 * a, use_bias=False)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Reshape((4, 4, a))(x)
    x = upsample_block(
        x,
        128,
        LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        64,
        LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    
    x = upsample_block(
        x, 
        1, 
        Activation("tanh"),
        strides=(4, 4), 
        use_bias=False, 
        use_bn=True
    )
    """
    # At this point, we have an output which has the same shape as the input, (32, 32, 1).
    # We will use a Cropping2D layer to make it (28, 28, 1).
    x = Cropping2D((2, 2))(x)
    """

    g_model = Model(noise, x, name="generator")
    return g_model





class evals_WGANGP(Model):
    def __init__(self):
        super(evals_WGANGP, self).__init__()
        
        self.critic = get_critic_model()
        self.generator = get_generator_model()
        self.latent_dim = latent_dim
        self.c_steps = critic_extra_steps
        self.gp_weight = gp_weight

    def compile(self):
        super(evals_WGANGP, self).compile()
        self.c_optimizer = critic_optimizer
        self.g_optimizer = generator_optimizer
        self.c_loss_fn = critic_loss
        self.g_loss_fn = generator_loss

    def gradient_penalty(self, batch_size, real_evals, fake_evals):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the critic loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_evals - real_evals
        interpolated = real_evals + alpha * diff

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

    def train_step(self, real_evals):
        if isinstance(real_evals, tuple):
            real_evals = real_evals[0]

        # Get the batch size
        batch_size = tf.shape(real_evals)[0]

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
                # Generate fake eigenvalues from the latent vector
                fake_evals = self.generator(random_latent_vectors, training=True)
                # Normalize the fake eigenvalues
                norm_fake_evals = normalize_evals(fake_evals)
                # Get the logits for the fake eigenvalues
                fake_logits = self.critic(norm_fake_evals, training=True)
                # Normalize the real eigenvalues
                norm_real_evals = normalize_evals(real_evals)
                # Get the logits for the real eigenvalues
                real_logits = self.critic(norm_real_evals, training=True)

                # Calculate the critic loss using the fake and real eigenvalues logits
                c_cost = self.c_loss_fn(real_evals=real_logits, fake_evals=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, norm_real_evals, norm_fake_evals)
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
            generated_evals = self.generator(random_latent_vectors, training=True)
            # Normalize generated eigenvalues
            norm_generated_evals = normalize_evals(generated_evals)   
            # Get the critic logits for fake images
            gen_evals_logits = self.critic(norm_generated_evals, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_evals_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"c_loss": c_loss, "g_loss": g_loss}

    def generate_evals(self, nb_trial=100):
        
        for i in range(nb_trial):
            random_latent_vectors = tf.random.normal(shape=(1, self.latent_dim))
            generated_evals = self.generator.predict(random_latent_vectors)
            norm_generated_evals = normalize_evals(generated_evals) # scale appropriately
            predictions = self.critic(norm_generated_evals)
            print(predictions[0, 0])
            
            if predictions[0, 0] > 0:
                print("real eigenvalues found")
                return True, norm_generated_evals
                
        
        print("no real eigenvalues found")
        return False, norm_generated_evals







"""
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
        # Because we have a critic instead of a critic, accuracy will always be zero otherwise  
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
        # Calculate the gradients for critic
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
"""
