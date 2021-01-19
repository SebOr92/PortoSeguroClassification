import pandas as pd
import numpy as np
import statistics
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA

def load_and_check_data(path):

    data = pd.read_csv(path,
                       sep=',')
    print("Summary of data")
    print("Shape of data: {shape}".format(shape=data.shape))
    print(data.head())
    print(data.tail())
    print(data.dtypes)
    print(data.columns)
    print("Counts of missing values per column: {miss}".format(miss = data.eq(-1).sum()))
    print("Distribution of negative target variable in data: {counts}".format(counts = data["target"].value_counts()[0]/len(data)))
    print("Successfully loaded data from CSV")

    return data

def group_data(data):

    # binary and categorical features are defined by their name. 
    binaries = [b for b in data.columns if re.search('_bin', str(b))]
    print(binaries)
    categoricals = [c for c in data.columns if re.search('_cat', str(c))]
    print(categoricals)
    
    # ordinal data is available as integers, contiuous data as floats
    ordinals = [o for o in data.columns if data[o].dtypes == 'int64' and o not in binaries and o not in categoricals]
    print(ordinals)
    continuous = [con for con in data.columns if data[con].dtypes == 'float64' and con not in binaries and con not in categoricals]
    print(continuous)

    return binaries, categoricals, ordinals, continuous

def handle_missing_values(data, remove):

    # fill missing data (-1 per kaggle definition) and drop rows containing missing data
    for c in continuous:
        data[c].replace(-1, np.NaN, inplace=True)
        data[[c]].fillna(data[c].mean(), inplace=True)

    for o in ordinals:
        data[o].replace(-1, np.NaN, inplace=True)
        data[[o]].fillna(data[o].mode(),inplace=True)

    for b in binaries:
        data[b].replace(-1, np.NaN, inplace=True)

    for c in categoricals:
        data[c].replace(-1, np.NaN, inplace=True)
    
    data.drop(remove, axis = 1, inplace=True)
    data.dropna(axis=0, inplace=True)
    print(data.shape)

    return data

def descriptive_statistics(data, binaries, categoricals, ordinals, continuous):
    print("Binary data")
    for b in binaries:
        print(data[b].value_counts())
    
    print("Categorical data")
    for c in categoricals:
        print(data[c].value_counts())
    
    print("Ordinal data")
    for o in ordinals:
        print(data[o].describe())
    
    print("Continuous data")
    for con in continuous:
        print(data[con].describe())
       
def split_data(data, seed, test_size, val_size):

    # drop unwanted columns and define target
    X = data.drop(['id', 'target'], axis=1)
    y = data["target"]

    # split data into training, testing and evaluation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_size, random_state=seed, stratify=y_test)
    print("X_train shape: {X} \n y_train shape: {y}".format(X=X_train.shape, y=y_train.shape))
    print("X_test shape: {X} \n y_test shape: {y}".format(X=X_test.shape, y=y_test.shape))
    print("X_val shape: {X} \n y_val shape: {y}".format(X=X_val.shape, y=y_val.shape))

    return X_train, X_test, X_val, y_train, y_test, y_val

def train_test_and_evaluate(seed, X_train, X_test, y_train, y_test, onehots, numericals, cv):

    # define transformer for features that are one hot encoded and pre-select features using chi square
    selector = SelectKBest(chi2, k=15)
    ohc = OneHotEncoder(handle_unknown='ignore')
    onehot_transformer = Pipeline(steps=[
        ('selector', selector),
        ('ohc', ohc)])

    # define transformer for numerical features, scale features and shrink space using pca
    scaler = StandardScaler()
    pca = PCA()
    numeric_transformer = Pipeline(steps=[
        ('scaler', scaler),
        ('pca', pca)])

    # define resampling: oversample minority class first and undersample majority class afterwards
    over = RandomOverSampler(random_state = random_seed, sampling_strategy=0.1)
    under = RandomUnderSampler(random_state = random_seed, sampling_strategy=0.5)
    
    # combine steps into preprocessing and machine learning pipeline using logistic regression
    preprocessor = ColumnTransformer(transformers=[
        ('numeric', numeric_transformer, numericals),
        ('onehot', onehot_transformer, onehots)])

    pipe_model = Pipeline(steps=[
        ('over', over),
        ('under', under),
        ('prep', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))])

    # Cross-validate model on training data to estimate performance
    cv_results = 2* roc_auc_score(y_train, cross_val_predict(pipe_model, X_train, y_train, cv=cv, method='predict_proba')[:,1]) - 1
    print("Mean training gini after CV: {res}".format(res = cv_results.mean()))

    # Fit the model to training data and evaluate on test data and finally on evaluation data
    pipe_model.fit(X_train, y_train)
    y_true, y_pred= y_test, pipe_model.predict(X_test)
    gini = 2* roc_auc_score(y_true, pipe_model.predict_proba(X_test)[:,1]) - 1
    print("Gini score on test set: " + str(gini))
    
    y_true, y_pred= y_val, pipe_model.predict(X_val)
    gini = 2* roc_auc_score(y_true, pipe_model.predict_proba(X_val)[:,1]) - 1
    print("Gini score on validation set: " + str(gini))

# load data
data = load_and_check_data('data.csv')

# set random state
random_seed = 191

# define feature types
binaries, categoricals, ordinals, continuous = group_data(data)

# remove data with too many missing values
remove = ['ps_reg_03', 'ps_car_03_cat', 'ps_car_05_cat']

id_and_target = ['id', 'target']
binaries = [x for x in binaries if x not in remove]
categoricals = [x for x in categoricals if x not in remove]
ordinals = [x for x in ordinals if x not in remove and x not in id_and_target]
continuous = [x for x in continuous if x not in remove and x not in id_and_target ]

# describe data
descriptive_statistics(data, binaries, categoricals, ordinals, continuous)

# remove remaining missing data
data = handle_missing_values(data, remove)

# split data
X_train, X_test, X_val, y_train, y_test, y_val = split_data(data, random_seed, 0.33, 0.1)

# define cv and features for one hot encoding
cv = StratifiedKFold(n_splits=5)
onehots = binaries + categoricals

#train and evaluate model
model = train_test_and_evaluate(random_seed, X_train, X_test, y_train, y_test, onehots, continuous, cv)
