from __future__ import print_function
import keras
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np


class AUCHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.model.validation_data[:-1])
        auc = roc_auc_score(self.model.validation_data[-1], y_pred)
        print("\nEpoch validation AUC: {}\n".format(auc))
