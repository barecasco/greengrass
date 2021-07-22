import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 30)
import eda_utils as et
from importlib import reload
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.utils import resample
reload(et)

# ================================================================= GENERATE TRAINING & TEST DATA
# explanation of the eda and feature engineering is commented on eda_utils file.
# some visualization was also done using some function defined in that file

main_df = pd.read_csv("dataset.csv")
df = et.clean_df_mk1(main_df.copy())
default_corr = df.corr()['Default']
sorted_corrlist = list(default_corr.abs().sort_values().index)

def generate_train_test(excluded_columns=[]):
    df = et.clean_df_mk1(main_df.copy())
    df = et.exclude_columns(df, excluded_columns)
    X = df[df.columns.values[:-1]].copy()
    y = df.iloc[:,-1].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # upsampling the training data for default 1
    # so the voting/mode in the leaf is not affected by the imbalanced class
    y_count = Counter(y_train)
    one_indices = np.array(y_train[y_train==1].index)
    zero_indices = np.array(y_train[y_train==0].index)
    one_indices = resample(one_indices, n_samples=y_count[0])
    upsampled_indices = list(one_indices) + list(zero_indices)

    X_train = X_train.loc[upsampled_indices,:]
    y_train = y_train[upsampled_indices]

    # the test data should also be balanced to see the performance fairly.
    # precision and recall are affected by the imbalance in support
    # the test data is balanced by downsampling
    y_count = Counter(y_test)
    zero_indices = np.array(y_test[y_test==0].index)
    all_indices = np.array(y_test.index)
    choices = np.random.choice(len(zero_indices), size=y_count[0]-y_count[1], replace=False)
    downsampled_indices = [i for i in all_indices if i not in zero_indices[choices]]

    X_test = X_test.loc[downsampled_indices,:]
    y_test = y_test[downsampled_indices]

    return X_train, X_test, y_train, y_test


# ================================================================= MODELLING THOUGHTS
# The final result of eda and feature engineering: most of the features are binaries.
# Three families of algorithm came in mind: the trees, naive bayes and logit.
# I choose the model by testing how they perform and see which one work in easiest manner while having a good performance.

# logit performance test was done
# naive bayes performance test was done
# trees (decision tree, random forest, adaptive boost, gradient boost) performance test was done

# the logit was having a hard time to converge. The model is too complicated.
# naive bayes works better but up to a limit.
# the tree needs to evolve up to its boosted version to work best, using adaptive boosting

# !!! the task now is to find any enhancement of adaboost and/or naive bayes in recent studies
# !!! I failed.

# Attempted to try stacking ensemble but meta_classifier (decisiontree) worked poorly

# After some comparison, adaboost is selected.

# shown below is the selected algorithm, adaptive boost of trees with selected hyperparameters

# ================================================================= ADAPTIVE BOOSTING
from sklearn.ensemble import AdaBoostClassifier

# gender has lowest correlation and excluded
X_train, X_test, y_train, y_test = generate_train_test(['gender'])
ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=8)
ada.fit(X_train, y_train)

y_test_pred = ada.predict(X_test)
print (classification_report(y_test, y_test_pred))

"""
AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=8)

              precision    recall  f1-score   support

           0       0.81      0.74      0.77       464
           1       0.76      0.83      0.79       464

    accuracy                           0.78       928
"""

# ---------------------------------------------------- store model
import pickle
filename = './models/ada.sav'
pickle.dump(ada, open(filename, 'wb'))
