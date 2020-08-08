import numpy as np
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest
import hdf5storage as hds

# load all the data for progression prediction
# the mat file contains the radiomics features, clinical features, and progression labels of patients
data_all = hds.loadmat('/path/to/feats_labels.mat')


# extract the train, validation, and test sets from the data file

# labels are in the form of vector
# the critical patients are marked with 1 and the non-critical ones are marked with 0.
labels_train = data_all['labels_train']
labels_val = data_all['labels_val']
labels_test = data_all['labels_test']

# the time from admission to critical illness (for critical patients)
# and the time from admission to discharge (for non-critical patients)
time_train = data_all['time_train']
time_val = data_all['time_val']
time_test = data_all['time_test']

# the radiomics features that are in the shape of num_of_patients * dimension_of_feature_vector
feat_train = data_all['feat_train']
feat_val = data_all['feat_val']
feat_test = data_all['Feat_Test']

# the clinical features that are in the shape of num_of_patients * dimension_of_feature_vector
clin_train = data_all['clin_train']
clin_val = data_all['clin_val']
clin_test = data_all['clin_test']


# convert the labels to the format of survival forest model
labels_crit_train = np.ndarray(shape=(clin_train.shape[0], ), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
labels_crit_val = np.ndarray(shape=(clin_val.shape[0], ), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
labels_crit_test = np.ndarray(shape=(clin_test.shape[0], ), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

for i in range(clin_train.shape[0]):
    if labels_train[i, 0] == 1:
        labels_crit_train[i] = (True,  time_train[i, 0])
    else:
        labels_crit_train[i] = (False, time_train[i, 0])

for i in range(clin_val.shape[0]):
    if labels_val[i, 0] == 1:
        labels_crit_val[i] = (True,  time_val[i, 0])
    else:
        labels_crit_val[i] = (False, time_val[i, 0])

for i in range(clin_test.shape[0]):
    if labels_test[i, 0] == 1:
        labels_crit_test[i] = (True,  time_test[i, 0])
    else:
        labels_crit_test[i] = (False, time_test[i, 0])


# the radiomics based prediction model
# define the parameters of forest model
random_state = 'num of state'
n_estimators = 'num of estimators'
max_depth = 'max depth'
min_samples_split = 'min_samples_split'
min_samples_leaf = 'min_samples_leaf'
max_features = 'max_features'
bootstrap = True

rsf = RandomSurvivalForest(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                           random_state=random_state, bootstrap=bootstrap,
                           min_samples_split=min_samples_split,
                           min_samples_leaf=min_samples_leaf)
rsf.fit(feat_train, labels_crit_train)


# the risk scores of each subject
scores_train_rad = rsf.predict(feat_train)
scores_test_rad = rsf.predict(feat_test)
scores_val_rad = rsf.predict(feat_val)

# the c_index of radiomics based prediction
c_ind_test_rad = rsf.score(feat_test, labels_crit_test)


# the clinical feature based prediction
random_state = 'num of state'
n_estimators = 'num of estimators'
max_depth = 'max depth'
min_samples_split = 'min_samples_split'
min_samples_leaf = 'min_samples_leaf'
max_features = 'max_features'
bootstrap = True

rsf = RandomSurvivalForest(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                           random_state=random_state, bootstrap=bootstrap,
                           min_samples_split=min_samples_split,
                           min_samples_leaf=min_samples_leaf)
rsf.fit(clin_train, labels_crit_train)


# the risk scores of each subject
scores_train_clin = rsf.predict(clin_train)
scores_test_clin = rsf.predict(clin_test)
scores_val_clin = rsf.predict(clin_val)

# the c_index of clinical based prediction
c_ind_test_clin = rsf.score(clin_test, labels_crit_test)

# use the weighted sum of radiomics based prediction and clinical based prediction as the combined prediction
# a and 1 - a are the balanced weights
a = 'a value in the range of [0, 1]'
scores_test_com = a * scores_test_rad + (1 - a) * scores_test_clin

# c_index of the combined prediction
c_ind_com = concordance_index_censored(labels_test.astype(bool).squeeze(), time_test.squeeze(), scores_test_com.squeeze())


