import preprocessing.preprocessing as prepro
import model.cnn as cnn
import tensorflow as tf
import keras

def main():
    dataset_train, dataset_test, CLASSES, NB_CLASSES = prepro.preprocessing()

    data_augmentation = prepro.get_data_augmentation()

    model = cnn.get_cnn_arch(256,256,data_augmentation)
    model.compile(
    optimizer = 'adam',
    loss = tf.losses.BinaryCrossentropy(),
    metrics = ['accuracy', cnn.f1_score, cnn.precision, cnn.recall])
    
    cnn.train_or_get_model(dataset_train, dataset_test, 50, model, '../model',cnn.get_tensorboard())
    cnn.evaluate_model(model, dataset_test)
    cnn.get_confusion_matrix_and_roc(model, dataset_test)
    
    keras.utils.plot_model(model, to_file='model.png' ,show_shapes=True, dpi=64)
    

if __name__ == '__main__':

    main()