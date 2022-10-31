import glob
import numpy as np
import pandas as pd
from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

import rcr

def save_features_as_csv():
    return

def load_features_csv_concat(folder_path):
    df_list = []
    for file in glob.glob(folder_path+"/*.csv"):
        df_ = pd.read_csv(file)
        df_list.append(df_)
    df = pd.concat(df_list)
    df = df.reset_index(drop=True)
    return df

def exclude_subject(df,exluded_subjects):
    condition_string = ''
    for ex_sub in exluded_subjects:
        condition_string += "(df['Subject'] !='" +ex_sub+"') & "
    evaluation_string = 'df['+condition_string[:len(condition_string)-2]+']'
    df_ex = eval(evaluation_string)
    return df_ex.reset_index(drop=True)

def electrode_wise_dataframe(df, condition_list, id_vars = ['Subject', 'Task', 'Electrode']):
    stats_frame = df[
        ['Subject', 'Task', 'Electrode','Lentr', 'TT', 'L', 'RR', 'LAM', 'DET', 'V','Vmax', 'Ventr', 'W','Wentr']
    ]

    stats_frame.melt(id_vars=id_vars, var_name='RQA_feature', value_name='feature_value')
    stats = stats_frame.pivot_table(index=['Subject', 'Task'], columns='Electrode',
                                     values=['Lentr', 'TT', 'L', 'RR', 'LAM', 'DET', 'V','Vmax', 'Ventr', 'W', 'Wentr']).reset_index()

    stats = stats.replace(condition_list[0], 0)
    stats = stats.replace(condition_list[1], 1)
    y = stats.Task.values
    return stats, y


def electrode_wise_dataframe_epochs(df, condition_list, id_vars = ['Subject', 'Task', 'Epoch_id','Electrode']):
    stats_frame = df[
        ['Subject', 'Task','Epoch_id','Electrode','Lentr', 'TT', 'L', 'RR', 'LAM', 'DET', 'V','Vmax', 'Ventr', 'W','Wentr']
    ]

    stats_frame.melt(id_vars=id_vars, var_name='RQA_feature', value_name='feature_value')
    stats = stats_frame.pivot_table(index=['Subject', 'Task'], columns=['Electrode', 'Epoch_id'],
                                    values=['Lentr', 'TT', 'L', 'RR', 'LAM', 'DET', 'V','Vmax', 'Ventr', 'W', 'Wentr']).reset_index()

    stats = stats.replace(condition_list[0], 0)
    stats = stats.replace(condition_list[1], 1)
    y = stats.Task.values
    return stats, y


def select_features_clean_and_normalize(df,features=['Lentr', 'TT', 'L', 'LAM', 'DET','V', 'Ventr', 'W','Wentr']):

    stats_data = df[features].values

    #rcr
    stats_data_cleaned=np.empty((stats_data.shape[0],stats_data.shape[1]))
    stats_data_cleaned[:]=np.nan
    r = rcr.RCR(rcr.SS_MEDIAN_DL)

    for ii in range(stats_data.shape[1]):
        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(16,8),dpi=200)
        # ax1.hist(stats_data[:,ii])
        # ax1.set_title('Raw')

        r.performBulkRejection(stats_data[:,ii])
        cleaned_data_indices = r.result.indices
        stats_data_cleaned[cleaned_data_indices,ii]=stats_data[cleaned_data_indices,ii]

        # ax2.hist(stats_data_cleaned[:,ii][~np.isnan(stats_data_cleaned[:,ii])])
        # ax2.set_title('Cleaned')

        # plt.savefig('Feature_nr_'+str(ii)+'jpg')
        # plt.close()


    df_stats_data_cleaned=pd.DataFrame(stats_data_cleaned)
    # df_stats_data_cleaned=df_stats_data_cleaned.fillna(method='mean', axis=0)#+df_stats_data_cleaned.fillna(method='bfill', axis=0))/2
    # df_stats_data_cleaned.interpolate(limit=5, inplace=True)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(df_stats_data_cleaned)

    stats_data_cleaned = imputer.transform(df_stats_data_cleaned)

    ####normalize#########
    stats_data_normed=np.empty((stats_data.shape[0],stats_data.shape[1]))
    for ii in range(stats_data.shape[1]):
        stats_data_normed[:,ii] = (stats_data_cleaned[:,ii]-stats_data_cleaned[:,ii].min(axis=0))/ (stats_data_cleaned[:,ii].max(axis=0)-stats_data_cleaned[:,ii].min(axis=0)) #stats_data[:,ii]-stats_data[:,ii].mean(axis=0))/ stats_data[:,ii].std(axis=0)

    return stats_data_normed


def clasyfication_SVM(df,y,cv=10,type='linear'):


    clf=svm.SVC(kernel=type)
    skf = StratifiedKFold(n_splits=cv)
    # run split() again to generate folds
    folds = skf.split(df, y)
    print('folds shape ', folds)
    performance = np.zeros(skf.n_splits)
    performance_open= np.zeros(skf.n_splits)
    performance_closed= np.zeros(skf.n_splits)

    for i, (train_idx, test_idx) in enumerate(folds):

        X_train = df[train_idx,:]
        y_train = y[train_idx]

        X_test = df[test_idx,:]
        y_test = y[test_idx]

        # call fit (on train) and predict (on test)
        model = clf.fit(X=X_train, y=y_train)
        y_hat = model.predict(X=X_test)

        # calculate accuracy
        performance[i] = accuracy_score(y_test, y_hat)
        cm = confusion_matrix(y_test, y_hat)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # class_acuracy = cm.diagonal()
        class_acuracy = cm.diagonal()
        performance_open[i]=class_acuracy[0]*100
        performance_closed[i]=class_acuracy[1]*100

    # calculate average accuracy
    print('Mean performance: %.3f' % np.mean(performance*100))
    print('Mean performance 1st class: %.3f' % np.mean(performance_open))
    print('Mean performance 2nd class: %.3f' % np.mean(performance_closed))


    lin = svm.SVC(kernel=type).fit(X_train, y_train)
    lin_pred = lin.predict(X_test)

    return lin, lin_pred

def cross_validation(df,y,cv=10,title = 'cv job',type='linear'):

    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel=type)
    # The "accuracy" scoring shows the proportion of correct classifications

    min_features_to_select = 4  # Minimum number of features to consider
    rfecv = RFECV(
        estimator=svc,
        step=1,
        cv=StratifiedKFold(n_splits=cv),
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
    )
    rfecv.fit(df, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(
        range(min_features_to_select, len(rfecv.cv_results_['mean_test_score']) + min_features_to_select),
        rfecv.cv_results_['mean_test_score'],
    )
    plt.title(title)
    plt.show()
    plt.savefig(title+' classification with feature selection_more_features'+str(rfecv.n_features_)+'_'+str(round(max(rfecv.cv_results_['mean_test_score'])*100,2))+'.png', dpi=150)
    plt.close()

    return rfecv.transform(df)



def compute_binary_SVM(df,y,predict_on_all_data = False,type='linear'):

    # stats_data = df[['TT', 'RR', 'DET', 'LAM', 'L', 'Lentr']].values
    X_train, X_test, y_train, y_test = model_selection.train_test_split(df, y, train_size=0.80, test_size=0.20,
                                                                        random_state=101)
    global lin

    if predict_on_all_data:
        print('SVM prediction on all data')
        lin = svm.SVC(kernel=type).fit(X_train, y_train)

        lin_pred = lin.predict(df)

        lin_accuracy = accuracy_score(y, lin_pred)

        print('Accuracy (Linear Kernel): ', "%.2f" % (lin_accuracy * 100))

        cm = confusion_matrix(y, lin_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        class_acuracy = cm.diagonal()
        print('Accuracy (1st class): ', "%.2f" % (class_acuracy[0] * 100))
        print('Accuracy (2nd class): ', "%.2f" % (class_acuracy[1] * 100))
    else:
        print('SVM prediction on test data')
        lin = svm.SVC(kernel=type).fit(X_train, y_train)

        lin_pred = lin.predict(X_test)

        lin_accuracy = accuracy_score(y_test, lin_pred)

        print('Accuracy (Linear Kernel): ', "%.2f" % (lin_accuracy * 100))
        print('Y train:', y_train)
        print('Y test:', y_test)
        print('Y pred:', lin_pred)

        cm = confusion_matrix(y_test, lin_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        class_acuracy = cm.diagonal()
        print('Accuracy (1st class): ', "%.2f" % (class_acuracy[0] * 100))
        print('Accuracy (2nd class): ', "%.2f" % (class_acuracy[1] * 100))

    return lin, lin_pred



def clasyfication_RFC(df,y,cv=10,max_depth=2):

    clf = RandomForestClassifier(max_depth=max_depth, random_state=0)
    skf = StratifiedKFold(n_splits=cv)
    # run split() again to generate folds
    folds = skf.split(df, y)

    performance = np.zeros(skf.n_splits)
    performance_open= np.zeros(skf.n_splits)
    performance_closed= np.zeros(skf.n_splits)

    for i, (train_idx, test_idx) in enumerate(folds):

        X_train = df[train_idx,:]
        y_train = y[train_idx]

        X_test = df[test_idx,:]
        y_test = y[test_idx]

        # call fit (on train) and predict (on test)
        model = clf.fit(X=X_train, y=y_train)
        y_hat = model.predict(X=X_test)

        # calculate accuracy
        performance[i] = accuracy_score(y_test, y_hat)
        cm = confusion_matrix(y_test, y_hat)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # class_acuracy = cm.diagonal()
        class_acuracy = cm.diagonal()
        performance_open[i]=class_acuracy[0]*100
        performance_closed[i]=class_acuracy[1]*100

    # calculate average accuracy
    print('Mean performance: %.3f' % np.mean(performance*100))
    print('Mean performance 1st class: %.3f' % np.mean(performance_open))
    print('Mean performance 2nd class: %.3f' % np.mean(performance_closed))


    lin = RandomForestClassifier(max_depth=max_depth, random_state=0)
    lin.fit(X=X_train, y=y_train)
    lin_pred = lin.predict(X_test)

    return lin, lin_pred






