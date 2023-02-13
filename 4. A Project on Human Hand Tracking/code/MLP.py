# %% [markdown]
# # Read data 

# %%
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import random
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from datetime import datetime
import gc
gc.collect()
torch.cuda.empty_cache()

random_seed = 1234
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2' # '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
print(device)

final_dict = {}


import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import random
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from datetime import datetime
import gc
gc.collect()
torch.cuda.empty_cache()

random_seed = 1234
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2' # '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
print(device)

final_dict = {}



# %%
X_train_ = torch.load('data/Training/X_train_diet.pt') # torch.Size([16008, 240, 320, 1]) # X_train_diet.pt # X_train.pt
X_test_ = torch.load('data/Testing/X_test_diet.pt') # torch.Size([1596, 240, 320, 1]) # X_test_diet.pt # X_test.pt

Y_train_ = pd.read_csv('data/Training/Y_train.csv')
Y_test_ = pd.read_csv('data/Testing/Y_test.csv')
# dataframe to tensor
Y_train_ = torch.tensor(Y_train_.values)  # torch.Size([16008, 63])
Y_test_ = torch.tensor(Y_test_.values)  # torch.Size([1596, 63])

# %%
X_train_ = X_train_.float()
X_test_ = X_test_.float()

mean = 1881.42
std = 12.29

# Standardise
X_train_ -= mean
X_train_ /= std
X_test_ -= mean
X_test_ /= std


# %%
# flatten image

B, H, W, C = X_train_.shape
X_train_ = X_train_.reshape(B, -1)  # torch.Size([16008, 76800])

B, H, W, C = X_test_.shape
X_test_ = X_test_.reshape(B, -1)  # torch.Size([1596, 76800])


# %%
def sample_and_split_dataloader(X_train_, X_test_, Y_train_, Y_test_, sample_size, valid_size, batch_size):

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
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_set, batch_size=batch_size, sampler=valid_sampler)

    # test loader
    test_set = TensorDataset(X_test_, Y_test_)
    test_loader = DataLoader(test_set, batch_size=1)

    # print shape
    print('\n Sample size: ', sample_size)
    print('train_loader: ', len(train_loader)*train_loader.batch_size )
    print('valid_loader: ', len(valid_loader)*valid_loader.batch_size )
    print('test_loader: ', len(test_loader)*test_loader.batch_size )

    return train_loader, valid_loader, test_loader


# %% [markdown]
# # Model

# %%
class my_MLP(nn.Module): 

    def __init__(self, hidden_size, input_size=76800, output_size=63, activation=nn.ELU()):
        super().__init__()
        self.HIDDEN_SIZE = hidden_size
        self.INPUT_SIZE = input_size
        self.OUTPUT_SIZE = output_size
        self.ACTIVATION = str(activation)
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        out = self.fc(x)
        return out

# %%
def train(model):
    
    optimizer = torch.optim.SGD(model.parameters(), lr = LR)
    criterion = nn.MSELoss()
    valid_loss_min = np.inf # 초기화 (나중에 업데이트 함)
    records = {}

    for epoch in tqdm(range(1, Train_epoch + 1)):

        train_loss = 0.0
        valid_loss = 0.0

        for batch_id, (image, label) in enumerate(train_loader): # iter: batch 데이터 (25개) 

            label, image = label.to(device), image.float().to(device) # shape: (25,)
            output = model(image)   # 1. 모델에 데이터 입력해 출력 얻기 # 10개 클래스에 대한 로짓 # shape: (25, 10)
            loss = criterion(output.float(), label.float()) # 2. loss 계산 
            train_loss += loss.item()
            
            optimizer.zero_grad() # 3. 기울기 초기화 (iter 끝날때마다 초기화)
            loss.backward() # 4. 역전파
            optimizer.step() # 5. 최적화
        
        for batch_id, (image, label) in enumerate(valid_loader):

            label, image = label.to(device), image.float().to(device)
            output = model(image)
            loss = criterion(output.float(), label.float())
            valid_loss += loss.item()
        
        # calculate avg losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

        # print training/validation records 
        #print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        # save training/validation records 
        records[f'epoch_{epoch}'] = [train_loss, valid_loss]
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            #print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(
                model, 
                f'./records/AccRun_MLP_best_model_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_HiddenSize_{model.HIDDEN_SIZE}_activation_{model.ACTIVATION}.pt')
            torch.save(
                model.state_dict(), 
                f'./records/AccRun_MLP_best_model_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_HiddenSize_{model.HIDDEN_SIZE}_activation_{model.ACTIVATION}.pth')
            valid_loss_min = valid_loss
    # save records
    pd.DataFrame(records).to_csv(
        f'./records/AccRun_MLP_training_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_HiddenSize_{model.HIDDEN_SIZE}_activation_{model.ACTIVATION}.csv', index=False)
    
    return model


def test(model):

    print('success load best_model')
    criterion = nn.MSELoss()

    test_loss = 0.0
    with torch.no_grad():  # 파라미터 업데이트 안 함

        for batch_id, (image, label) in enumerate(tqdm(test_loader)):

            label, image = label.to(device), image.float().to(device)
            output = model(image)
            loss = criterion(output.float(), label.float())
            test_loss += loss.item()

    # calculate avg losses
    test_loss = test_loss/len(test_loader.dataset)

    return test_loss


# %% [markdown]
# # Ablation Study: Dropout

# %%
class my_MLP_dropout(nn.Module):

    def __init__(self, hidden_size, input_size=76800, output_size=63, activation=nn.ELU(), dropout=nn.Dropout(0.2)):
        super().__init__()
        self.HIDDEN_SIZE = hidden_size
        self.INPUT_SIZE = input_size
        self.OUTPUT_SIZE = output_size
        self.ACTIVATION = str(activation)
        self.dropout = dropout

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            self.dropout,
            nn.Linear(hidden_size, hidden_size),
            activation,
            self.dropout,
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        out = self.fc(x)
        return out


# %%
def train_dropout(model):
    
    optimizer = torch.optim.SGD(model.parameters(), lr = LR)
    criterion = nn.MSELoss()
    valid_loss_min = np.inf # 초기화 (나중에 업데이트 함)
    records = {}

    for epoch in tqdm(range(1, Train_epoch + 1)):

        train_loss = 0.0
        valid_loss = 0.0

        for batch_id, (image, label) in enumerate(train_loader): # iter: batch 데이터 (25개) 

            label, image = label.to(device), image.float().to(device) # shape: (25,)
            output = model(image)   # 1. 모델에 데이터 입력해 출력 얻기 # 10개 클래스에 대한 로짓 # shape: (25, 10)
            loss = criterion(output.float(), label.float()) # 2. loss 계산 
            train_loss += loss.item()
            
            optimizer.zero_grad() # 3. 기울기 초기화 (iter 끝날때마다 초기화)
            loss.backward() # 4. 역전파
            optimizer.step() # 5. 최적화
        
        for batch_id, (image, label) in enumerate(valid_loader):

            label, image = label.to(device), image.float().to(device)
            output = model(image)
            loss = criterion(output.float(), label.float())
            valid_loss += loss.item()
        
        # calculate avg losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

        # print training/validation records 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        # save training/validation records 
        records[f'epoch_{epoch}'] = [train_loss, valid_loss]
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(
                model, 
                f'./records/MLP_dropout_best_model_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_HiddenSize_{model.HIDDEN_SIZE}_activation_{model.ACTIVATION}.pt')
            torch.save(
                model.state_dict(), 
                f'./records/MLP_dropout_best_model_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_HiddenSize_{model.HIDDEN_SIZE}_activation_{model.ACTIVATION}.pth')
            valid_loss_min = valid_loss
    # save records
    pd.DataFrame(records).to_csv(
        f'./records/MLP_dropout_training_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_HiddenSize_{model.HIDDEN_SIZE}_activation_{model.ACTIVATION}.csv', index=False)
    
    return model


def test_dropout(model):

    print('success load best_model')
    criterion = nn.MSELoss()

    test_loss = 0.0
    with torch.no_grad():  # 파라미터 업데이트 안 함

        for batch_id, (image, label) in enumerate(tqdm(test_loader)):

            label, image = label.to(device), image.float().to(device)
            output = model(image)
            loss = criterion(output.float(), label.float())
            test_loss += loss.item()

    # calculate avg losses
    test_loss = test_loss/len(test_loader.dataset)

    return test_loss


# %% [markdown]
# # Ablation Study: LayerNorm

# %%
class my_MLP_layernorm(nn.Module):

    def __init__(self, hidden_size, input_size=76800, output_size=63, activation=nn.ELU()):
        super().__init__()
        self.HIDDEN_SIZE = hidden_size
        self.INPUT_SIZE = input_size
        self.OUTPUT_SIZE = output_size
        self.ACTIVATION = str(activation)

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            activation,
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size)
        )

    def forward(self, x):
        out = self.fc(x)
        return out


# %%
def train_layernorm(model):

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    valid_loss_min = np.inf  # 초기화 (나중에 업데이트 함)
    records = {}

    for epoch in tqdm(range(1, Train_epoch + 1)):

        train_loss = 0.0
        valid_loss = 0.0

        for batch_id, (image, label) in enumerate(train_loader):  # iter: batch 데이터 (25개)

            label, image = label.to(device), image.float().to(
                device)  # shape: (25,)
            # 1. 모델에 데이터 입력해 출력 얻기 # 10개 클래스에 대한 로짓 # shape: (25, 10)
            output = model(image)
            loss = criterion(output.float(), label.float())  # 2. loss 계산
            train_loss += loss.item()

            optimizer.zero_grad()  # 3. 기울기 초기화 (iter 끝날때마다 초기화)
            loss.backward()  # 4. 역전파
            optimizer.step()  # 5. 최적화

        for batch_id, (image, label) in enumerate(valid_loader):

            label, image = label.to(device), image.float().to(device)
            output = model(image)
            loss = criterion(output.float(), label.float())
            valid_loss += loss.item()

        # calculate avg losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

        # print training/validation records
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        # save training/validation records
        records[f'epoch_{epoch}'] = [train_loss, valid_loss]
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))
            torch.save(
                model,
                f'./records/MLP_layernorm_best_model_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_HiddenSize_{model.HIDDEN_SIZE}_activation_{model.ACTIVATION}.pt')
            torch.save(
                model.state_dict(),
                f'./records/MLP_layernorm_best_model_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_HiddenSize_{model.HIDDEN_SIZE}_activation_{model.ACTIVATION}.pth')
            valid_loss_min = valid_loss
    # save records
    pd.DataFrame(records).to_csv(
        f'./records/MLP_layernorm_training_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_HiddenSize_{model.HIDDEN_SIZE}_activation_{model.ACTIVATION}.csv', index=False)

    return model


def test_layernorm(model):

    print('success load best_model')
    criterion = nn.MSELoss()

    test_loss = 0.0
    with torch.no_grad():  # 파라미터 업데이트 안 함

        for batch_id, (image, label) in enumerate(tqdm(test_loader)):

            label, image = label.to(device), image.float().to(device)
            output = model(image)
            loss = criterion(output.float(), label.float())
            test_loss += loss.item()

    # calculate avg losses
    test_loss = test_loss/len(test_loader.dataset)

    return test_loss


# %% [markdown]
# # Main

# %%
# training parameters

n_class = 63
Train_epoch = 100
BATCH_SIZE = 128
LR = 5e-2



acc_runtime_dict = {}

# "optimized" hyperparameters
param_grid = {
    'HIDDEN_SIZE': [130],
    'ACTIVATION': [nn.Sigmoid()]
}

grid_list = list(ParameterGrid(param_grid))

# main

if __name__ == '__main__':

    ################ train model #####################

    sample_size_list = [0.3, 0.6, 1.0]

    for sample_size in tqdm(sample_size_list):

        train_loader, valid_loader, test_loader = sample_and_split_dataloader(
            X_train_, X_test_, Y_train_, Y_test_, sample_size=sample_size, valid_size=0.2, batch_size=BATCH_SIZE)

        acc_runtime_dict = {}

        for case in tqdm(grid_list):

            print(case)

            # try:
            regr = my_MLP(
                hidden_size=case['HIDDEN_SIZE'], activation=case['ACTIVATION']).to(device)

            start_time = datetime.now()
            model_trained = train(regr)
            train_time = datetime.now() - start_time

            acc_runtime_dict[str(case)] = dict({'train_time': train_time})
            # except:
            #     pass

        final_dict[sample_size] = acc_runtime_dict

    ################ save results #####################

    # save
    with open('result_MLP.pickle', 'wb') as f:
        pickle.dump(final_dict, f, pickle.HIGHEST_PROTOCOL)

    # load
    with open('result_MLP.pickle', 'rb') as f:
        result_RF = pickle.load(f)

    ################# final test ######################

    train_loader, valid_loader, test_loader = sample_and_split_dataloader(
        X_train_, X_test_, Y_train_, Y_test_, sample_size=1.0, valid_size=0.2, batch_size=128)

    best_model = torch.load(
        './records/AccRun_MLP_best_model_TrainSampleSize_12928_HiddenSize_130_activation_Sigmoid().pt')

    start_time = datetime.now()
    test_loss = test(best_model)
    test_time = datetime.now() - start_time

    print('test_time: ', test_time.seconds)
    print('test_loss: ', test_loss)



