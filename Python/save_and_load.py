
# coding: utf-8

# ##### Copyright 2019 The TensorFlow Authors.

# In[1]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# In[2]:


#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# # Save and load models

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/keras/save_and_load"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_load.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_load.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/keras/save_and_load.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# Model progress can be saved during and after training. This means a model can resume where it left off and avoid long training times. Saving also means you can share your model and others can recreate your work. When publishing research models and techniques, most machine learning practitioners share:
# 
# * code to create the model, and
# * the trained weights, or parameters, for the model
# 
# Sharing this data helps others understand how the model works and try it themselves with new data.
# 
# Caution: TensorFlow models are code and it is important to be careful with untrusted code. See [Using TensorFlow Securely](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md) for details.
# 
# ### Options
# 
# There are different ways to save TensorFlow models depending on the API you're using. This guide uses [tf.keras](https://www.tensorflow.org/guide/keras)—a high-level API to build and train models in TensorFlow. The new, high-level `.keras` format used in this tutorial is recommended for saving Keras objects, as it provides robust, efficient name-based saving that is often easier to debug than low-level or legacy formats. For more advanced saving or serialization workflows, especially those involving custom objects, please refer to the [Save and load Keras models guide](https://www.tensorflow.org/guide/keras/save_and_serialize). For other approaches, refer to the [Using the SavedModel format guide](../../guide/saved_model.ipynb).

# ## Setup
# 
# ### Installs and imports

# Install and import TensorFlow and dependencies:

# In[3]:


get_ipython().system('pip install pyyaml h5py  # Required to save models in HDF5 format')


# In[4]:


import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)


# ### Get an example dataset
# 
# To demonstrate how to save and load weights, you'll use the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). To speed up these runs, use the first 1000 examples:

# In[5]:


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# ### Define a model

# Start by building a simple sequential model:

# In[6]:


# Define a simple sequential model
def create_model():
  model = tf.keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()


# ## Save checkpoints during training

# You can use a trained model without having to retrain it, or pick-up training where you left off in case the training process was interrupted. The `tf.keras.callbacks.ModelCheckpoint` callback allows you to continually save the model both *during* and at *the end* of training.
# 
# ### Checkpoint callback usage
# 
# Create a `tf.keras.callbacks.ModelCheckpoint` callback that saves weights only during training:

# In[7]:


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.


# This creates a single collection of TensorFlow checkpoint files that are updated at the end of each epoch:

# In[8]:


os.listdir(checkpoint_dir)


# As long as two models share the same architecture you can share weights between them. So, when restoring a model from weights-only, create a model with the same architecture as the original model and then set its weights. 
# 
# Now rebuild a fresh, untrained model and evaluate it on the test set. An untrained model will perform at chance levels (~10% accuracy):

# In[9]:


# Create a basic model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))


# Then load the weights from the checkpoint and re-evaluate:

# In[10]:


# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# ### Checkpoint callback options
# 
# The callback provides several options to provide unique names for checkpoints and adjust the checkpointing frequency.
# 
# Train a new model, and save uniquely named checkpoints once every five epochs:

# In[11]:


# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Calculate the number of batches per epoch
import math
n_batches = len(train_images) / batch_size
n_batches = math.ceil(n_batches)    # round up the number of batches to the nearest whole integer

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*n_batches)

# Create a new model instance
model = create_model()

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images, 
          train_labels,
          epochs=50, 
          batch_size=batch_size, 
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)


# Now, review the resulting checkpoints and choose the latest one:

# In[12]:


os.listdir(checkpoint_dir)


# In[13]:


latest = tf.train.latest_checkpoint(checkpoint_dir)
latest


# Note: The default TensorFlow format only saves the 5 most recent checkpoints.
# 
# To test, reset the model, and load the latest checkpoint:

# In[14]:


# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# ## What are these files?

# The above code stores the weights to a collection of [checkpoint](../../guide/checkpoint.ipynb)-formatted files that contain only the trained weights in a binary format. Checkpoints contain:
# * One or more shards that contain your model's weights.
# * An index file that indicates which weights are stored in which shard.
# 
# If you are training a model on a single machine, you'll have one shard with the suffix: `.data-00000-of-00001`

# ## Manually save weights
# 
# To save weights manually, use `tf.keras.Model.save_weights`. By default, `tf.keras`—and the `Model.save_weights` method in particular—uses the TensorFlow [Checkpoint](../../guide/checkpoint.ipynb) format with a `.ckpt` extension. To save in the HDF5 format with a `.h5` extension, refer to the [Save and load models](https://www.tensorflow.org/guide/keras/save_and_serialize) guide.

# In[15]:


# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# ## Save the entire model
# 
# Call `tf.keras.Model.save` to save a model's architecture, weights, and training configuration in a single `model.keras` zip archive.
# 
# An entire model can be saved in three different file formats (the new `.keras` format and two legacy formats: `SavedModel`, and `HDF5`). Saving a model as `path/to/model.keras` automatically saves in the latest format.
# 
# **Note:** For Keras objects it's recommended to use the new high-level `.keras` format for richer, name-based saving and reloading, which is easier to debug. The low-level SavedModel format and legacy H5 format continue to be supported for existing code.
# 
# You can switch to the SavedModel format by:
# 
# - Passing `save_format='tf'` to `save()`
# - Passing a filename without an extension
# 
# You can switch to the H5 format by:
# - Passing `save_format='h5'` to `save()`
# - Passing a filename that ends in `.h5`
# 
# Saving a fully-functional model is very useful—you can load them in TensorFlow.js ([Saved Model](https://www.tensorflow.org/js/tutorials/conversion/import_saved_model), [HDF5](https://www.tensorflow.org/js/tutorials/conversion/import_keras)) and then train and run them in web browsers, or convert them to run on mobile devices using TensorFlow Lite ([Saved Model](https://www.tensorflow.org/lite/models/convert/#convert_a_savedmodel_recommended_), [HDF5](https://www.tensorflow.org/lite/models/convert/#convert_a_keras_model_))
# 
# \*Custom objects (for example, subclassed models or layers) require special attention when saving and loading. Refer to the **Saving custom objects** section below.

# ### New high-level `.keras` format

# The new Keras v3 saving format, marked by the `.keras` extension, is a more
# simple, efficient format that implements name-based saving, ensuring what you load is exactly what you saved, from Python's perspective. This makes debugging much easier, and it is the recommended format for Keras.
# 
# The section below illustrates how to save and restore the model in the `.keras` format.

# In[16]:


# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a `.keras` zip archive.
model.save('my_model.keras')


# Reload a fresh Keras model from the `.keras` zip archive:

# In[17]:


new_model = tf.keras.models.load_model('my_model.keras')

# Show the model architecture
new_model.summary()


# Try running evaluate and predict with the loaded model:

# In[18]:


# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)


# ### SavedModel format

# The SavedModel format is another way to serialize models. Models saved in this format can be restored using `tf.keras.models.load_model` and are compatible with TensorFlow Serving. The [SavedModel guide](../../guide/saved_model.ipynb) goes into detail about how to `serve/inspect` the SavedModel. The section below illustrates the steps to save and restore the model.

# In[19]:


# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
get_ipython().system('mkdir -p saved_model')
model.save('saved_model/my_model') 


# The SavedModel format is a directory containing a protobuf binary and a TensorFlow checkpoint. Inspect the saved model directory:

# In[20]:


# my_model directory
get_ipython().system('ls saved_model')

# Contains an assets folder, saved_model.pb, and variables folder.
get_ipython().system('ls saved_model/my_model')


# Reload a fresh Keras model from the saved model:

# In[21]:


new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()


# The restored model is compiled with the same arguments as the original model. Try running evaluate and predict with the loaded model:

# In[22]:


# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)
