import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import umap
import umap.plot
from umap.parametric_umap import ParametricUMAP
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from mne.datasets import eegbci

def compute_Linear_Kernel(df):
    stats_frame = df[
        ['Subject', 'Task', 'Electrode', 'Lentr', 'TT', 'L', 'RR', 'LAM', 'DET']
    ]

    stats_frame.melt(id_vars=['Subject', 'Task', 'Electrode'], var_name='RQA_feature', value_name='feature_value')
    stats1 = stats_frame.pivot_table(index=['Subject', 'Task'], columns='Electrode',
                                     values=['TT', 'RR', 'DET', 'LAM', 'L', 'Lentr']).reset_index()

    y = stats1.Task.values
    stats_data = stats1[['TT', 'RR', 'DET', 'LAM', 'L', 'Lentr']].values
    X_train, X_test, y_train, y_test = model_selection.train_test_split(stats_data, y, train_size=0.80, test_size=0.20,
                                                                        random_state=101)

    lin = svm.SVC(kernel='linear').fit(X_train, y_train)

    lin_pred = lin.predict(X_test)

    lin_accuracy = accuracy_score(y_test, lin_pred)

    print('Accuracy (Linear Kernel): ', "%.2f" % (lin_accuracy * 100))

    cm = confusion_matrix(y_test, lin_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    class_acuracy = cm.diagonal()
    print('Accuracy (open): ', "%.2f" % (class_acuracy[0] * 100))
    print('Accuracy (close): ', "%.2f" % (class_acuracy[1] * 100))