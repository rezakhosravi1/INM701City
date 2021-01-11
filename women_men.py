# %%
import tensorflow as tf 
import tensorflow_probability as tfp 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, MobileNetV2
import numpy as np
import os
import shutil
from pathlib2 import Path
import pickle
import cv2
import matplotlib.pyplot as plt 
# %%
tfk = tf.keras
tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
# %%

root_path = Path('./')
data_path = root_path / 'archive/data'
classes = ['men', 'women']
test_ratio = 0.2

for c in classes:
    
    os.makedirs( root_path / 'train'/ c)
    os.makedirs( root_path / 'test' / c)

    source_path = data_path / c
    file_names = os.listdir(source_path)
    np.random.shuffle(file_names)
    test_name, train_name = np.split(np.array(file_names), [int(len(file_names)*test_ratio)])

    train_split = [source_path / name for name in train_name.tolist()]
    test_split = [source_path / name  for name in test_name.tolist()]

    for name in train_split:
        shutil.copy(name,  root_path / 'train'/ c )
    
    for name in test_split:
        shutil.copy(name, root_path / 'test'/ c)

print('Image splitting is done!')

# %%
# preprocessing and augmenting and generating the input images

image_size = (128, 128)
batch_size = 32

trian_data_generator_args = dict(horizontal_flip=True,
                                 rescale=1./255.0,
                                 rotation_range=40,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 fill_mode='nearest')
train_datagen = ImageDataGenerator(**trian_data_generator_args)
train_generator = train_datagen.flow_from_directory(
    './train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',)

test_datagen = ImageDataGenerator(rescale=1./255.0)
test_grenerator = test_datagen.flow_from_directory(
    './test',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary')

# %%
# BUILD THE MODEL 
base_model = VGG16(include_top=False,
                  weights='imagenet',
                  input_shape=(image_size[0], image_size[1], 3))
                
base_model.trainable = False

input_image = tfk.Input(shape=(image_size[0], image_size[1], 3), name='original_image')
output_features = base_model(input_image)
output_average = tfkl.GlobalAveragePooling2D()(output_features)
output_fc_1 = tfkl.Dense(128, activation=tf.nn.leaky_relu, name='fc_1')(output_average)
output = tfkl.Dense(1, activation='sigmoid', name='output')(output_fc_1)
model = tfk.Model(inputs=input_image, outputs=output, name='my_model')

model.summary()

initial_learning_rate = 0.0001
optimizer = tfk.optimizers.Adam(learning_rate=initial_learning_rate)
loss = tfk.losses.BinaryCrossentropy()
metrics = ['accuracy']
num_epochs = 10

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.fit(train_generator, validation_data=test_grenerator, epochs=num_epochs)
# %%
base_model = VGG16(include_top=False,
                  weights='imagenet',
                  input_shape=(image_size[0], image_size[1], 3))
                
base_model.trainable = False
for layer in base_model.layers:
    if layer.name == in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
        layer.trainable = True


input_image = tfk.Input(shape=(image_size[0], image_size[1], 3), name='original_image')
output_features = base_model(input_image)
output_average = tfkl.GlobalAveragePooling2D()(output_features)
output_fc_1 = tfkl.Dense(128, activation=tf.nn.leaky_relu, name='fc_1')(output_average)
output = tfkl.Dense(1, activation='sigmoid', name='output')(output_fc_1)
model = tfk.Model(inputs=input_image, outputs=output, name='my_model')

model.summary()


initial_learning_rate = 0.0001
optimizer = tfk.optimizers.Adam(learning_rate=initial_learning_rate)
loss = tfk.losses.BinaryCrossentropy()
metrics = ['accuracy']
num_epochs = 10

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.fit(train_generator, validation_data=test_grenerator, epochs=num_epochs)

# %%
base_model = VGG16(include_top=False,
                  weights='imagenet',
                  input_shape=(image_size[0], image_size[1], 3))
                
base_model.trainable = False
for layer in base_model.layers:
    if layer.name == in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
        layer.trainable = True

input_image = tfk.Input(shape=(image_size[0], image_size[1], 3), name='original_image')
output_features = base_model(input_image)
output_average = tfkl.GlobalAveragePooling2D()(output_features)
output_fc_1 = tfkl.Dense(128, activation=tf.nn.leaky_relu, name='fc_1')(output_average)
output = tfkl.Dense(1, activation='sigmoid', name='output')(output_fc_1)
model = tfk.Model(inputs=input_image, outputs=output, name='model_with_vgg16_as_base')

model.summary()


initial_learning_rate = 0.0003
learning_rate_schedule = tfk.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=40, decay_rate=.3, staircase=True
    )

checkpoint_cb = tfk.callbacks.ModelCheckpoint(
    'vgg16_cp.h5', monitor='val_accuracy',
    mode='max', save_best_only=True, verbose=2
)

early_stopping_cb = tfk.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.001, patience=5, 
    mode='auto', restore_best_weights=True
)

optimizer = tfk.optimizers.Adam(learning_rate=initial_learning_rate)
loss = tfk.losses.BinaryCrossentropy()
metrics = ['accuracy']
num_epochs = 10

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
vgg16_history = model.fit(train_generator, validation_data=test_grenerator, epochs=num_epochs)
model.save('mw_vgg16_as_base.h5')

with open('./vgg16_history', 'wb') as history:
    pickle.dump(vgg16_history.history, history)
# %%
base_model = MobileNetV2(include_top=False,
                          weights='imagenet',
                          input_shape=(image_size[0], image_size[1], 3))
                
base_model.trainable = False
for layer in base_model.layers[100:]:
        layer.trainable = True

input_image = tfk.Input(shape=(image_size[0], image_size[1], 3), name='original_image')
output_features = base_model(input_image)
output_average = tfkl.GlobalAveragePooling2D()(output_features)
output_fc_1 = tfkl.Dense(128, activation=tf.nn.leaky_relu, name='fc_1')(output_average)
output = tfkl.Dense(1, activation='sigmoid', name='output')(output_fc_1)
model = tfk.Model(inputs=input_image, outputs=output, name='model_with_mobile_net_v2_as_base')

model.summary()

initial_learning_rate = 0.0003
learning_rate_schedule = tfk.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=40, decay_rate=.3, staircase=True
    )

checkpoint_cb = tfk.callbacks.ModelCheckpoint(
    'mobile_net_v2_cp.h5', monitor='val_accuracy',
    mode='max', save_best_only=True, verbose=2
)

early_stopping_cb = tfk.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.001, patience=5, 
    mode='auto', restore_best_weights=True
)

optimizer = tfk.optimizers.Adam(learning_rate=initial_learning_rate)
loss = tfk.losses.BinaryCrossentropy()
metrics = ['accuracy']
num_epochs = 10

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
mobile_net_v2_history = model.fit(train_generator, validation_data=test_grenerator, epochs=num_epochs)
model.save('mw_mobilenetv2_as_base.h5')

with open('./mobiel_net_v2_history', 'wb') as history:
    pickle.dump(mobile_net_v2_history.history, history)
# %%
file_input = open('mobiel_net_v2_history', 'rb')
mobile_net_v2_history = pickle.load(file_input)
mobile_net_v2_history.keys()
acc = mobile_net_v2_history['accuracy']
val_acc = mobile_net_v2_history['val_accuracy']

loss = mobile_net_v2_history['loss']
val_loss = mobile_net_v2_history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy on MobileNetV2 as the base model')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss on MobileNetV2 as the base model')
plt.xlabel('epoch')
plt.show()
# %%
file_input = open('vgg16_history', 'rb')
vgg16_history = pickle.load(file_input)
vgg16_history.keys()
acc = vgg16_history['accuracy']
val_acc = vgg16_history['val_accuracy']

loss = vgg16_history['loss']
val_loss = vgg16_history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy on VGG16 as the base model')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss on VGG16 as the base model')
plt.xlabel('epoch')
plt.show()

# %%
image_size = (128, 128)
batch_size = 32
latent_dim = 32

class Sampling(tfkl.Layer):

    def call(self, inputs):
        z_mean, z_log_var = tf.split(inputs, num_or_size_splits=2, axis=-1)
        epsilon = tfk.backend.random_normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z_mean, z_log_var, z

base_model = MobileNetV2(include_top=False,
                          weights='imagenet',
                          input_shape=(image_size[0], image_size[1], 3))
                
base_model.trainable = False
for layer in base_model.layers[100:]:
        layer.trainable = True

# ENCODER
input_image = tfk.Input(shape=(image_size[0], image_size[1], 3), name='input_image')
output_features = base_model(input_image)
output_average = tfkl.GlobalAveragePooling2D()(output_features)
output_fc_1 = tfkl.Dense(128, activation=tf.nn.leaky_relu, name='fc_1')(output_average)
output_fc_2 = tfkl.Dense(latent_dim + latent_dim, activation=tfk.activations.linear, name='output')(output_fc_1)
z_mean, z_log_var, z = Sampling()(output_fc_2)
men_encoder = tfk.Model(inputs=input_image, 
                    outputs=[z_mean,
                             z_log_var,
                             z], 
                    name='men_encoder'
                    )

# GENERATOR/DECODER
latent_input = tfk.Input(shape=(latent_dim,), name='z_sampling')
input_reshaped = tfkl.Reshape(target_shape=(1,1,latent_dim))(latent_input)
output = tfkl.Conv2DTranspose(filters=256, kernel_size=4, strides=1, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation=None, padding='valid', 
                              name='convT_1')(input_reshaped)
output = tfkl.Conv2D(filters=256, kernel_size=4, strides=1, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                     activation=None, padding='same', 
                     name='d_conv_1')(output)
output = tfkl.BatchNormalization()(output)
output = tfkl.LeakyReLU(alpha=0.2)(output)
output = tfkl.Conv2DTranspose(filters=128, kernel_size=5, strides=1,kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation=None, padding='valid', 
                              name='convT_2')(output)
output = tfkl.Conv2D(filters=128, kernel_size=5, strides=1,kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                     activation=None, padding='same', 
                     name='d_conv_2')(output)
output = tfkl.BatchNormalization()(output)
output = tfkl.LeakyReLU(alpha=0.2)(output)
output = tfkl.Conv2DTranspose(filters=64, kernel_size=3, strides=2, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation=None, padding='same',
                              name='convT_3')(output)
output = tfkl.Conv2D(filters=64, kernel_size=3, strides=1, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation=None, padding='same',
                              name='d_conv_3')(output)
output = tfkl.BatchNormalization()(output)
output = tfkl.LeakyReLU(alpha=0.2)(output)
output = tfkl.Conv2DTranspose(filters=64, kernel_size=3, strides=2, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation=None, padding='same',
                              name='convT_4')(output)
output = tfkl.Conv2D(filters=64, kernel_size=3, strides=1, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation=None, padding='same',
                              name='d_conv_4')(output)
output = tfkl.BatchNormalization()(output)
output = tfkl.Dropout(.2)(output)
output = tfkl.Conv2DTranspose(filters=32, kernel_size=3, strides=2, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation=None, padding='same',
                              name='convT_5')(output)
output = tfkl.Conv2D(filters=32, kernel_size=3, strides=1, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation=None, padding='same',
                              name='d_conv_5')(output)
output = tfkl.BatchNormalization()(output)
output = tfkl.Dropout(.2)(output)
output = tfkl.Conv2DTranspose(filters=32, kernel_size=3, strides=2, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation=None, padding='same',
                              name='convT_6')(output)
output = tfkl.Conv2D(filters=32, kernel_size=3, strides=1, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation=None, padding='same',
                              name='d_conv_6')(output)
output = tfkl.BatchNormalization()(output)
output = tfkl.Dropout(.2)(output)
output = tfkl.BatchNormalization()(output)
fake_image = tfkl.Conv2D(filters=3, kernel_size=2, strides=1, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                                  padding='same', activation=tfk.activations.linear)(output) 
men_generator = tfk.Model(inputs=latent_input,
                          outputs=fake_image,
                          name='men_generator')
men_generator.summary()
# %%
# ANOTHER NETWORK 
image_size = (128, 128)
batch_size = 32
latent_dim = 2

class Sampling(tfkl.Layer):

    def call(self, inputs):
        z_mean, z_log_var = tf.split(inputs, num_or_size_splits=2, axis=-1)
        epsilon = tfk.backend.random_normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z_mean, z_log_var, z

base_model = VGG16(include_top=False,
                  weights='imagenet',
                  input_shape=(image_size[0], image_size[1], 3))
                
base_model.trainable = False
for layer in base_model.layers:
    if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
        layer.trainable = True


# base_model = MobileNetV2(include_top=False,
#                           weights='imagenet',
#                           input_shape=(image_size[0], image_size[1], 3))
                
# base_model.trainable = False
# for layer in base_model.layers[100:]:
#         layer.trainable = True

# ENCODER
input_image = tfk.Input(shape=(image_size[0], image_size[1], 3), name='input_image')
output_features = base_model(input_image)
output_average = tfkl.GlobalAveragePooling2D()(output_features)
output_fc_1 = tfkl.Dense(128, activation=tf.nn.leaky_relu, name='fc_1')(output_average)
output_fc_2 = tfkl.Dense(latent_dim + latent_dim, name='output')(output_fc_1)
z_mean, z_log_var, z = Sampling()(output_fc_2)
men_encoder = tfk.Model(inputs=input_image, 
                        outputs=[z_mean,
                                 z_log_var,
                                 z], 
                        name='men_encoder')
men_encoder.summary()

# GENERATOR/DECODER
latent_input = tfk.Input(shape=(latent_dim,), name='z_sampling')
dense_layer = tfkl.Dense(4*4*latent_dim, activation='relu', name='d_fc_1')(latent_input)
input_reshaped = tfkl.Reshape(target_shape=(4,4,latent_dim))(dense_layer)
output = tfkl.Conv2DTranspose(filters=512, kernel_size=3, strides=2, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation='relu', padding='same', 
                              name='convT_1')(input_reshaped)
output = tfkl.Conv2D(filters=512, kernel_size=3, strides=1, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                     activation='relu', padding='same', 
                     name='d_conv_1')(output)
output = tfkl.BatchNormalization()(output)
output = tfkl.Conv2DTranspose(filters=512, kernel_size=3, strides=2, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation='relu', padding='same', 
                              name='convT_2')(output)
output = tfkl.Conv2D(filters=512, kernel_size=3, strides=1, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                     activation='relu', padding='same', 
                     name='d_conv_2')(output)
output = tfkl.BatchNormalization()(output)
output = tfkl.Conv2DTranspose(filters=256, kernel_size=3, strides=2,kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation=None, padding='same', 
                              name='convT_3')(output)
output = tfkl.Conv2D(filters=256, kernel_size=3, strides=1,kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                     activation='relu', padding='same', 
                     name='d_conv_3')(output)
output = tfkl.BatchNormalization()(output)
output = tfkl.Dropout(.5)(output)
output = tfkl.Conv2DTranspose(filters=128, kernel_size=3, strides=2, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation='relu', padding='same',
                              name='convT_4')(output)
output = tfkl.Conv2D(filters=64, kernel_size=3, strides=1, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation='relu', padding='same',
                              name='d_conv_4')(output)
output = tfkl.BatchNormalization()(output)
output = tfkl.Conv2DTranspose(filters=64, kernel_size=3, strides=2, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation='relu', padding='same',
                              name='convT_5')(output)
output = tfkl.Conv2D(filters=64, kernel_size=3, strides=1, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                              activation='relu', padding='same',
                              name='d_conv_5')(output)
output = tfkl.BatchNormalization()(output)
output = tfkl.Dropout(.5)(output)
fake_image = tfkl.Conv2D(filters=3, kernel_size=3, strides=1, kernel_regularizer=tfk.regularizers.L1L2(0.01,0.01),
                                 activation='sigmoid', padding='same')(output) 
men_generator = tfk.Model(inputs=latent_input,
                          outputs=fake_image,
                          name='men_generator')
men_generator.summary()
# %%
@tf.function
def log_normal_pdf(z, z_mean, z_log_var):
    return tf.reduce_sum(
        -0.5 * ((z - z_mean)) ** 2. * tf.exp(-z_log_var) + z_log_var + tf.math.log(2.* np.pi),
        axis=-1)

# DEFINE THE METRIC(S) WE WISH TO TRACK AND WRITE THE SUMMARY
loss_tracker = tfk.metrics.Mean(name='loss')

class VAEMen(tfk.Model):

    def __init__(self, encoder, generator, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    
    @tf.function
    def train_step(self, data):

        with tf.GradientTape() as tape:
            loss = self.loss_calc(data)
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients_clipped = [tf.clip_by_norm(g, 5.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients_clipped, self.trainable_variables))

        # TRACK METRICS
        loss_tracker.update_state(loss)
        return {'loss': loss_tracker.result()}
    
    @tf.function
    def test_step(self, data):

        with tf.GradientTape() as tape:
            loss = self.loss_calc(data)

        # TRACK METRICS
        loss_tracker.update_state(loss)
        return {'val_loss': loss_tracker.result()}


    @tf.function
    def call(self, inputs, training=True):
        if training:
            self.train_step(inputs)
        else:
            self.test_step(inputs)

    @property
    def metrics(self,):
        # TO RESET METRICS AT THE START OF EACH EPOCH
        return [loss_tracker,]

    @tf.function
    def loss_calc(self, inputs):
        # image, labels = inputs 
        z_mean, z_log_var, z = self.encoder(inputs)
        likelihood = self.generator(z)
        log_likelihood_loss =  -tf.reduce_sum(self.loss_fn(labels=inputs, logits=likelihood), axis=[1,2,3])
        log_true_prior_density_distribution_loss = log_normal_pdf(z, 0.0, 0.0)
        log_posterior_conditional_density_distribution_loss = log_normal_pdf(z,
                                                                             z_mean,
                                                                             z_log_var)
        # MINIMIZING THE NEGATIVE ELBO IS LIKE MAXIMIZING THE ELBO
        # WE USE MONTE CARLO ESTIMATION OF ELBO
        return -tf.reduce_mean(log_likelihood_loss + \
                               log_true_prior_density_distribution_loss - \
                               log_posterior_conditional_density_distribution_loss)


    @tf.function
    def sample(self, z=None):
        if z is None:
            z = tf.random.normal(shape=(32,self.latent_dim))
        return self.generator(z, training=False)

    @tf.function
    def generate_from_test(self, inputs):
        _,_,z = self.encoder(inputs)
        return self.generator(z)
# %%
image_size = (128, 128)
batch_size = 32
seed = 1
trian_data_generator_args = dict(horizontal_flip=True,
                                 rescale=1./255.0,)
                                #  rotation_range=40,
                                #  width_shift_range=0.1,
                                #  height_shift_range=0.1,
                                #  shear_range=0.2,
                                #  zoom_range=0.2,
                                #  fill_mode='nearest')
train_image_datagen_men = ImageDataGenerator(**trian_data_generator_args)
# WE SHOULD MAKE A COPY OF MEN FOLDER AND GIVE THE PATH OF PARENT FOLDER
# WITH ALL TRAIN AND TEST DATA
train_image_generator_men = train_image_datagen_men.flow_from_directory(
    './vae_men',
    target_size=image_size,
    batch_size=batch_size,
    class_mode=None,
    seed=seed)

initial_learning_rate = 0.0001
learning_rate_schedule = tfk.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10, decay_rate=0.85, staircase=True
    )

checkpoint_cb = tfk.callbacks.ModelCheckpoint(
    'vae_men_cp.h5', monitor='loss',
    mode='min', save_best_only=True, verbose=2
)

early_stopping_cb = tfk.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.001, patience=5, 
    mode='auto', restore_best_weights=True
)

optimizer = tfk.optimizers.Adam(learning_rate=initial_learning_rate)
loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
num_epochs = 150

vae_men = VAEMen(men_encoder, men_generator, latent_dim)
vae_men.compile(optimizer=optimizer, loss_fn=loss_fn)
vae_men_history = vae_men.fit(train_image_generator_men, epochs=num_epochs)

vae_men.save_weights('vae_men_model.h5')

with open('./vae_men_history', 'wb') as history:
    pickle.dump(vae_men_history.history, history)

# %%
# PLOT AUGMENTATED SAMPLE IMAGE 
image = cv2.imread('./train/men/01(1).jpg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(img)
img_batch = tf.expand_dims(img, 0)
data_augmentation = tfk.Sequential([
    tfkl.experimental.preprocessing.RandomZoom(.2),
    tfkl.experimental.preprocessing.RandomRotation(.3)
])
for i in range(4):
    augmented_image = data_augmentation(img_batch)
    ax = plt.subplot(2,2, i+1)
    plt.imshow(augmented_image[0])
    plt.axis('off')
# %%
# PLOT THE SAMPLE OUTPUT OF VAE DECODER
images = vae_men.sample()
imgs = images[:9]
plt.figure(figsize=(10,10))
for i in range(imgs.shape[0]):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(imgs[i])