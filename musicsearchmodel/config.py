#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
import tensorflow as tf
import os
# path to training and testing data
TRAIN_DATASET = "D:\\Documents\\Python scripts\\Music Retrieval through humming\\Spectrograms_Mapped"
TEST_DATASET = "D:\\Documents\\Python scripts\\Music Retrieval through humming\\Spectrograms_Mapped"
# model input image size
IMAGE_SIZE = (224, 224)
# batch size and the buffer size
BATCH_SIZE = 128
BUFFER_SIZE = BATCH_SIZE * 2
# define autotune
AUTO = tf.data.experimental.AUTOTUNE
# define the training parameters
LEARNING_RATE = 0.001
STEPS_PER_EPOCH = 35
VALIDATION_STEPS = 10
EPOCHS = 30
# define the path to save the model
OUTPUT_PATH = "output"
MODEL_PATH = os.path.join(OUTPUT_PATH, "siamese_network")
OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_PATH, "output_image.png")


# In[ ]:




