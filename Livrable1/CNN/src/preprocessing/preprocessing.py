import os
import tensorflow as tf
import keras
import keras.layers as layers


def get_dataframe(IMAGE_HEIGHT,IMAGE_WIDTH,BATCH_SIZE,VALIDATION_SPLIT,PATH_DATASET):
    dataset_train = tf.keras.preprocessing.image_dataset_from_directory (
    PATH_DATASET,
    image_size = (IMAGE_HEIGHT, IMAGE_WIDTH),
    validation_split = VALIDATION_SPLIT,
    batch_size = BATCH_SIZE,
    seed = 42,
    subset = 'training',)

    dataset_test = tf.keras.preprocessing.image_dataset_from_directory (
        PATH_DATASET,
        image_size = (IMAGE_HEIGHT, IMAGE_WIDTH),
        validation_split = VALIDATION_SPLIT,
        batch_size = BATCH_SIZE,
        seed = 42,
        subset = 'validation',)

    CLASSES = dataset_train.class_names
    NB_CLASSES = len(dataset_train.class_names)

    print(f'Le dataset comporte {NB_CLASSES} classes : {CLASSES}')

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset_train = dataset_train.cache().prefetch(buffer_size=AUTOTUNE)
    dataset_test = dataset_test.cache().prefetch(buffer_size=AUTOTUNE)

    return dataset_train, dataset_test, CLASSES, NB_CLASSES

def get_data_augmentation():
    data_augmentation = keras.Sequential([
    layers.RandomFlip(mode='horizontal'),
    layers.RandomRotation(0.2, fill_mode='reflect',
                          interpolation='bilinear', fill_value=0.0),
    layers.RandomZoom(0.2),])
    return data_augmentation

def preprocessing():
    BATCH_SIZE = 16
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    VALIDATION_SPLIT = 0.2

    PATH_DATASET = os.path.join('../dataset/train_val')

    dataset_train, dataset_test, CLASSES, NB_CLASSES = get_dataframe(IMAGE_HEIGHT,IMAGE_WIDTH,BATCH_SIZE,VALIDATION_SPLIT,PATH_DATASET)

    return dataset_train, dataset_test, CLASSES, NB_CLASSES