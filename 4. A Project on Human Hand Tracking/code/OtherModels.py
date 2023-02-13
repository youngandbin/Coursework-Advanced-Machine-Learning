# %%
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel, RBF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from datetime import datetime
SEED = 2022
np.random.seed(SEED)


# %%
# torch.Size([16008, 240, 320, 1])
X_train_ = torch.load('data/Training/X_train_diet.pt')
# torch.Size([1596, 240, 320, 1])
X_test_ = torch.load('data/Testing/X_test_diet.pt')

Y_train_ = pd.read_csv('data/Training/Y_train.csv')
Y_test_ = pd.read_csv('data/Testing/Y_test.csv')

# dataframe to tensor
Y_train_ = torch.tensor(Y_train_.values)  # torch.Size([16008, 63])
Y_test_ = torch.tensor(Y_test_.values)  # torch.Size([1596, 63])

# flatten image

B, H, W, C = X_train_.shape
X_train_ = X_train_.reshape(B, -1)  # torch.Size([16008, 76800])

B, H, W, C = X_test_.shape
X_test_ = X_test_.reshape(B, -1)  # torch.Size([1596, 76800])

def sample_and_split_dataset(X_train_, X_test_, Y_train_, Y_test_, sample_size, valid_size):

    # sample train set
    n_train = len(X_train_)
    indices = list(range(n_train))
    np.random.shuffle(indices)
    split = int(np.floor(sample_size * n_train))
    sample_indices = indices[:split]
    X_train_ = X_train_[sample_indices]
    Y_train_ = Y_train_[sample_indices]

    # split train, valid set
    n_train = len(X_train_)                  
    indices = list(range(n_train))
    np.random.shuffle(indices)

    split = int(np.floor(valid_size * n_train)) 
    train_idx, valid_idx = indices[split:], indices[:split]

    train_set = TensorDataset(X_train_, Y_train_)
    train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx) 
    train_loader = DataLoader(train_set, batch_size=len(train_sampler), sampler=train_sampler)
    valid_loader = DataLoader(train_set, batch_size=len(valid_sampler), sampler=valid_sampler)

    # save final data
    for img, labels in train_loader:
        X_train = img
        Y_train = labels
    for img, labels in valid_loader:
        X_valid = img
        Y_valid = labels
    X_test = X_test_
    Y_test = Y_test_

    # print shape
    print('\n Sample size: ', sample_size)
    print('X_train: ', X_train.shape)
    print('X_valid: ', X_valid.shape)
    print('X_test: ', X_test.shape)
    print('Y_train: ', Y_train.shape)
    print('Y_valid: ', Y_valid.shape)
    print('Y_test: ', Y_test.shape)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test




# %% [markdown]
# ## SVR 

# %%
final_dict = {}
acc_runtime_dict = {}  

# "optimized" hyperparameters

param_grid = {
    'kernel': ['linear'],
    'C' : [1.0]
  }

grid_list = list(ParameterGrid(param_grid))

# %%
sample_size_list = [1.0]

for sample_size in tqdm(sample_size_list):

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = sample_and_split_dataset(
        X_train_, X_test_, Y_train_, Y_test_, sample_size=sample_size, valid_size=0.2)

    acc_runtime_dict = {}

    for case in tqdm(grid_list):

        # try:
        regr = MultiOutputRegressor(
            SVR(kernel=case['kernel'], C=case['C'])
            )

        start_time = datetime.now()
        regr.fit(X_train, Y_train)
        train_time = datetime.now() - start_time

        start_time = datetime.now()
        pred = regr.predict(X_valid)
        valid_time = datetime.now() - start_time
        valid_error = mean_squared_error(Y_valid, pred)

        start_time = datetime.now()
        pred = regr.predict(X_test)
        test_time = datetime.now() - start_time
        test_error = mean_squared_error(Y_test, pred)

        acc_runtime_dict[str(case)] = dict({'train_time': train_time,
                                            'valid_time': valid_time, 'valid_error': valid_error,
                                            'test_time': test_time, 'test_error': test_error})
        # except:
        #     pass

    final_dict[sample_size] = acc_runtime_dict
    
# 51분에 1/6 돌아감 (full data)

# %%
# # save
with open('result_SVR.pickle', 'wb') as f:
    pickle.dump(final_dict, f, pickle.HIGHEST_PROTOCOL)

# load
with open('result_SVR.pickle', 'rb') as f:
    result_RF = pickle.load(f)


# %%
result_RF


# %% [markdown]
# ## Extra Trees 

# %%
final_dict = {}
acc_runtime_dict = {}

# "optimized" hyperparameters
param_grid = {
    'n_estimators': [150],
    'min_samples_split': [5],            
    'max_depth': [None],                  
    'max_features': ['sqrt']
}

grid_list = list(ParameterGrid(param_grid))


# %%
sample_size_list = [0.1, 0.2, 0.3]

for sample_size in tqdm(sample_size_list):

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = sample_and_split_dataset(
        X_train_, X_test_, Y_train_, Y_test_, sample_size=sample_size, valid_size=0.2)

    acc_runtime_dict = {}

    for case in tqdm(grid_list):

        # try:
        regr = ExtraTreesRegressor(
                n_estimators=case['n_estimators'], 
                min_samples_split=case['min_samples_split'],
                max_depth=case['max_depth'], 
                max_features=case['max_features'], random_state=0
            )

        start_time = datetime.now()
        regr.fit(X_train, Y_train)
        train_time = datetime.now() - start_time

        start_time = datetime.now()
        pred = regr.predict(X_valid)
        valid_time = datetime.now() - start_time
        valid_error = mean_squared_error(Y_valid, pred)

        start_time = datetime.now()
        pred = regr.predict(X_test)
        test_time = datetime.now() - start_time
        test_error = mean_squared_error(Y_test, pred)

        acc_runtime_dict[str(case)] = dict({'train_time': train_time,
                                            'valid_time': valid_time, 'valid_error': valid_error,
                                            'test_time': test_time, 'test_error': test_error})
        # except:
        #     pass

    final_dict[sample_size] = acc_runtime_dict

# 8분

# %%
# # save
with open('result_ExtraTrees.pickle', 'wb') as f:
    pickle.dump(final_dict, f, pickle.HIGHEST_PROTOCOL)

# load
with open('result_ExtraTrees.pickle', 'rb') as f:
    result_RF = pickle.load(f)


# %%
result_RF



