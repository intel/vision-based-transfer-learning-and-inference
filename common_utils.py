import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers

import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import models
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, GlobalAveragePooling2D

from timeit import default_timer as timer
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report,accuracy_score, top_k_accuracy_score

import argparse
from distutils.util import strtobool
import random

#Setting Global Seed for reproducibilty
random.seed(123)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

RANDOM_SEED = 123
IMG_SIZE = (224, 224)
tf_dict = {
  "colorectal": "colorectal_histology",
}

DISABLE_TF32 = True
precision = 'FP32'

class TransferLearning(object):
    """
    class for Transfer Learning Training
    """
    def __init__(self, inf_args):
        self.BATCH_SIZE = inf_args.BATCH_SIZE
        if(inf_args.DATASET_DIR):
            print("Dataset directory is ",inf_args.DATASET_DIR)
            self.dataset_dir = inf_args.DATASET_DIR
        else:
            self.dataset_dir = 'datasets/'
        self.train_dir = self.dataset_dir + '/train' # Setting Dataset Directory if dataset needs to be loaded from directory
        self.validation_dir = self.dataset_dir + '/val'
        self.test_dir = self.dataset_dir +  '/test'
        if(inf_args.OUTPUT_DIR):
            self.base_log_dir = inf_args.OUTPUT_DIR
            print("Setting Output log Directory")
        else:
            self.base_log_dir = "logs/fit" # Setting log Directoy
        os. makedirs(self.base_log_dir, exist_ok=True)
        self.feature_extractor_model = "https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/1" #Using Resnetv1.5 pretrained model
        self.total_classes = 0
        self.actual_labels = []
        self.num_epochs = inf_args.NUM_EPOCHS
        self.inference = inf_args.inference
        
    def resize_images(self, image, label):
        """
        Resizing Images to size 224 X 224
        """
        image = tf.image.resize(image, size=(224,224))
        image = tf.cast(image, dtype=tf.float32)
        return image, label
    
    
    def load_dataset_from_directory(self):
        """
        Loading Dataset from Directory
        """
        train_set = tf.keras.utils.image_dataset_from_directory(self.train_dir,
                                                            shuffle=True,
                                                            image_size=IMG_SIZE,batch_size=self.BATCH_SIZE)

        valid_set = tf.keras.utils.image_dataset_from_directory(self.validation_dir,
                                                                 shuffle=True,
                                                                 image_size=IMG_SIZE,batch_size=self.BATCH_SIZE)

        try:
            test_set = tf.keras.utils.image_dataset_from_directory(self.test_dir,
                                                                 shuffle=False,
                                                                 image_size=IMG_SIZE,batch_size=self.BATCH_SIZE)
        except:
            print("Since test directory files are not present so using validation files as test files")
            test_set = tf.keras.utils.image_dataset_from_directory(self.validation_dir,
                                                                 shuffle=False,
                                                                 image_size=IMG_SIZE,batch_size=self.BATCH_SIZE)
        #Updating total number of classes
        self.class_names = train_set.class_names
        self.total_classes = len(self.class_names)
        print(len(test_set))
        self.actual_labels = np.concatenate([y for x, y in test_set], axis=0)
        #Resizing Images
        train_set = train_set.map(map_func=self.resize_images, num_parallel_calls=tf.data.AUTOTUNE)
        self.train_dataset = train_set.shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.AUTOTUNE)

        valid_set = valid_set.map(map_func=self.resize_images, num_parallel_calls=tf.data.AUTOTUNE)
        self.validation_dataset = valid_set.prefetch(buffer_size=tf.data.AUTOTUNE)

        test_set = test_set.map(map_func=self.resize_images, num_parallel_calls=tf.data.AUTOTUNE)
        self.test_dataset = test_set.prefetch(buffer_size=tf.data.AUTOTUNE)
        
    def load_dataset_from_tfds(self, dataset_name='colorectal'):
        """
        Loading Dataset from TensforFlow dataset , cached files are used from second time
        """
        (train_set, valid_set), info = tfds.load(tf_dict[dataset_name],
                                   split=["train[0%:80%]", "train[80%:]"], 
                                   data_dir=self.dataset_dir,
                                   as_supervised=True, with_info=True)
        self.total_classes = len(info.features["label"].names)
        print("Since test directory files are not present so using validation files as test files")
        test_set = valid_set
        self.actual_labels = [y for x, y in test_set] 
        train_set = train_set.map(map_func=self.resize_images, num_parallel_calls=tf.data.AUTOTUNE)
        self.train_dataset = train_set.shuffle(buffer_size=1000).batch(self.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

        valid_set = valid_set.map(map_func=self.resize_images, num_parallel_calls=tf.data.AUTOTUNE)
        self.validation_dataset = valid_set.batch(self.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

        test_set = test_set.map(map_func=self.resize_images, num_parallel_calls=tf.data.AUTOTUNE)
        self.test_dataset = test_set.batch(self.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    def apply_augmentation(self):
        data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal_and_vertical",
                            input_shape=(224,224,3),seed=RANDOM_SEED),layers.RandomRotation(0.5, seed=RANDOM_SEED)])
        self.train_dataset = self.train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=tf.data.AUTOTUNE)
        
    def normalize(self):
        """
        Normalizing the Images within the range [0,1]
        """
        print("Normalizing")
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        self.train_dataset = self.train_dataset.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
        self.validation_dataset = self.validation_dataset.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
        self.test_dataset = self.test_dataset.map(lambda x, y: (normalization_layer(x), y))
        
    def get_total_classes(self):
        """
        Returns the total number of classes in the dataset
        """
        return self.total_classes
    
    def make_model(self, add_data_augmentation=False, add_denselayer=False):
        """
        Creates a model for Training , Use the feature vector from TF Hub and add on a dense classificationlayer based on the number of 
        classed in the dataset. Add data augmentation and few more dense layers for better accuracy. This is especially necessary if more
        classes are there.
        """
        data_augmentation = tf.keras.Sequential( # Augmentation Layer
                   [
                    layers.RandomFlip("horizontal_and_vertical",
                      input_shape=(224, 224,3),
                      seed=RANDOM_SEED),
                    layers.RandomRotation(0.5, seed=RANDOM_SEED),
                    layers.RandomZoom(0.3, seed=RANDOM_SEED),
                    ]
        )
        feature_extractor_layer = hub.KerasLayer(
            self.feature_extractor_model,
            input_shape=(224, 224, 3),
            trainable=False,dtype=tf.float32) # Trainable is set to False, as we do only finetuning
        model = models.Sequential()
        if(add_data_augmentation):
            model.add(data_augmentation)
        model.add(feature_extractor_layer)
        if(add_denselayer): # Adding more dense layers for better accuracy
            model.add(Dense(units=1024, activation='relu'))
            model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=self.total_classes, activation='softmax')) #Last classsification layer
        self.model = model
        
    def train_model(self, lr=0.001, num_epochs=100):
        """
        Train model for given number of epochs or till convergence.
        HyperParameters
        Optimizer : Adam
        Loss : SparseCategoricalCrossentropy
        """
        optimizer = tf.keras.optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['acc'])
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.base_log_dir, save_weights_only=False, monitor='val_acc',
                                    mode='max', save_best_only=True)
        # Stopping earling if convergence is reached 
        stop_early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        lr_decay = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    verbose=2,
                    mode='auto',
                    cooldown=1,
                    min_lr=0.0000000001,
                )
        start = timer()
        self.history = self.model.fit(self.train_dataset,
                    validation_data=self.validation_dataset,
                    verbose=2,
                    epochs=self.num_epochs,
                    initial_epoch=0,
                    callbacks=[model_checkpoint_callback, lr_decay, stop_early_callback]
                    ) 

        end = timer()
        print("Total elapsed Training time = ", end - start)
        print("Maximum validation accuracy = ", np.max(self.history.history['val_acc']))
       
    def evaluate(self,checkpoint_file):
        """
        Evaluates the model on accuracy metric and prints Top1 and Top5 Accuracy 
        """
        model = tf.keras.models.load_model(checkpoint_file)
        if(self.inference == True):
            history_filename = os.path.join(checkpoint_file, 'hist.npy')
            self.history_data = np.load(str(history_filename), allow_pickle='TRUE').item()
        else:
            self.history_data = self.history.history
        start = timer()
        loss, accuracy = model.evaluate(self.test_dataset)
        end = timer()
        print('Accuracy of model on test dataset:', accuracy)
        print("Total elapsed Test time = ", end - start)
        acc = self.history_data['acc']
        val_acc = self.history_data['val_acc']
        loss = self.history_data['loss']
        val_loss = self.history_data['val_loss']

        plt.figure(figsize=(12, 18))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        #plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        #plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        #plt.show()
        if (self.inference != True):
            plt.savefig(os.path.join(checkpoint_file, 'train_val_plot.png'))
            history_filename = os.path.join(checkpoint_file, 'hist.npy')
            with open(history_filename, 'wb') as f:
                np.save(f, self.history.history)
        predicted_values = model.predict(self.test_dataset)
        predicted_labels = np.argmax(predicted_values, axis=1)
        report = classification_report(self.actual_labels, predicted_labels)
        print("Classification report")
        print(report)
       # print("Top 1 accuracy score: ", top_k_accuracy_score(self.actual_labels, predicted_values, k=1))
       # print("Top 5 accuracy score: ", top_k_accuracy_score(self.actual_labels, predicted_values, k=5))
        
        
def setting_precision(inf_args):
    DISABLE_TF32 = True
    if(inf_args.precision == "Mixed_Precision"):
        print("Setting mixed precision")
        DISABLE_TF32 = False
        if(inf_args.platform != "SPR"):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            precision = 'FP16'
        else:
            tf.config.optimizer.set_experimental_options({'auto_mixed_precision_mkl':True})
            precision = 'BF16'
    if(DISABLE_TF32):
        tf.config.experimental.enable_tensor_float_32_execution(False)
        print("Is Tf32 enabled ? : ", tf.config.experimental.tensor_float_32_execution_enabled())