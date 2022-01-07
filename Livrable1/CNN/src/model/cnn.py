from sklearn import metrics
import tensorflow as tf 
import keras.layers as layers
import keras.backend as kb
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import roc_curve,RocCurveDisplay,auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import datetime

def get_cnn_arch(IMAGE_HEIGHT, IMAGE_WIDTH, data_augmentation):
    model = tf.keras.Sequential([
    #data_augmentation,

    layers.Rescaling(
        1./255, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    ])

    return model


def get_tensorboard():
    log_dir = "./CNN/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback

def train_or_get_model(dataset_train, dataset_test, EPOCHS, model, PATH_SAVE_MODEL, tensorboard_callback, newModel=True):
    if newModel == True:
        model_history = model.fit(dataset_train,steps_per_epoch=500, validation_data = dataset_test, epochs =EPOCHS, callbacks=[tensorboard_callback])
        statistics(model_history)
        keras.models.save_model(model, PATH_SAVE_MODEL)
        return model
    else :
        return keras.models.load_model(PATH_SAVE_MODEL, custom_objects={'f1_score':f1_score, 'precision':precision, 'recall':recall})   


########## METRICS ##########

def precision(y_true, y_pred):
    true_positives = kb.sum(kb.round(kb.clip(y_true * y_pred, 0, 1)))
    predicted_positives = kb.sum(kb.round(kb.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + kb.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = kb.sum(kb.round(kb.clip(y_true * y_pred, 0, 1)))
    possible_positives = kb.sum(kb.round(kb.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + kb.epsilon())
    return recall

# f1 score
def f1_score(y_true, y_pred):
    true_positives = kb.sum(kb.round(kb.clip(y_true * y_pred, 0, 1)))
    predicted_positives = kb.sum(kb.round(kb.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + kb.epsilon())
    possible_positives = kb.sum(kb.round(kb.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + kb.epsilon())
    f1_score = 2*(precision*recall)/(precision+recall+kb.epsilon())
    return f1_score


########## STATISTIC AND PREDICTION ##########

def statistics(model_save, EPOCHS) :
    acc = model_save.history['accuracy']
    val_acc = model_save.history['val_accuracy']

    loss = model_save.history['loss']
    val_loss = model_save.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.ylim([0, 1])
    plt.subplot(1, 2, 1)
    plt.plot(range(EPOCHS), acc, label='Training Accuracy')
    plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.ylim([0, 1])
    plt.plot(range(EPOCHS), loss, label='Training Loss')
    plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('../statistics/statistics.png')


def evaluate_model(model, dataset_test):
    model_evaluation = model.evaluate(dataset_test)


    print('Test loss:', model_evaluation[0])
    print('Test accuracy:', model_evaluation[1])
    print('Test F1 Score :', model_evaluation[2])
    print('Test Precision :', model_evaluation[3])
    print('Test Recall :', model_evaluation[4])

    fig, ax = plt.subplots(1, 1)
    column_labels = ["$\\bf{Loss}$",
                    "$\\bf{Accuracy}$",
                    "$\\bf{F1Score}$",
                    "$\\bf{Precision}$",
                    "$\\bf{Recall}$"]
    ax.axis('tight')
    ax.axis('off')
    plt.title('Scores for test dataset', fontsize=15)
    table = ax.table(cellText=[['%.3f %%' % (
        i*100) for i in model_evaluation]], colLabels=column_labels, loc="center")
    table.scale(2.5, 4)
    plt.savefig('../statistics/score.png')



def transformPrediction(n):
    return 0 if n < 0.5 else 1


# Partie de génération de matrice de confusion
def get_confusion_matrix_and_roc(model, dataset_test):

    preds = model.predict(dataset_test)
    predicted_categories = list(map(transformPrediction, preds))
    true_categories = tf.concat([y for x, y in dataset_test], axis=0)
    con_mat = confusion_matrix(predicted_categories, true_categories)
    disp = ConfusionMatrixDisplay(confusion_matrix=con_mat)
    disp.plot(cmap="Blues")
    plt.title('Confusion Matrix')
    plt.savefig('../statistics/confusion.png')

    fpr, tpr, _ = roc_curve(true_categories, predicted_categories)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    display.plot()
    plt.title('ROC Curve')
    plt.ylim(0,)
    plt.savefig('../statistics/roc.png')