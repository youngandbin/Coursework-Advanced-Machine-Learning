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
# permute data

X_train_ = X_train_.permute([0, 3, 1, 2])
X_train_.shape

X_test_ = X_test_.permute([0, 3, 1, 2])
X_test_.shape

# torch.Size([1596, 1, 240, 320])


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

    # # sample test set
    # n_test = len(X_test_)
    # indices = list(range(n_test))
    # np.random.shuffle(indices)
    # split = int(np.floor(sample_size * n_test))
    # sample_indices = indices[:split]
    # X_test_ = X_test_[sample_indices]
    # Y_test_ = Y_test_[sample_indices]

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
class my_CNN(nn.Module): 

    def __init__(self, n_class, CHANNELS, KERNEL_SIZE, STRIDE, PADDING):
        # super(my_CNN, self).__init__()
        super().__init__()
        self.n_class = n_class
        self.cs = CHANNELS
        self.KERNEL_SIZE = KERNEL_SIZE
        self.STRIDE = STRIDE
        self.PADDING = PADDING
        # self.dropout - nn.Dropout(0.4)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.cs[0], kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        ) 
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.cs[0], self.cs[1], kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(self.cs[1], self.cs[2], kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        
        self.fc = None

    def forward(self, x):
        
        out1 = self.layer1(x) 
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = out3.reshape(out3.size(0), -1) 
        ###
        shape1, shape2 = out3.shape[2], out3.shape[3]
        self.fc = nn.Linear(shape1 * shape2 * self.cs[2], self.n_class).to(device)
        ###
        out4 = out4.to(device)
        out = self.fc(out4)  #######################################################

        # print(x.shape)  # torch.Size([1, 1, 240, 320])
        # print(out1.shape)  # torch.Size([1, 16, 120, 160])
        # print(out2.shape)  # torch.Size([1, 32, 120, 160])
        # print(out3.shape)  # torch.Size([1, 64, 60, 80])
        # print(out4.shape) # torch.Size([1, 307200])

        return out


# MatMul shape 맞는지 체크

# a = my_CNN(n_class=63, CHANNELS=[6,6,16], KERNEL_SIZE=3, STRIDE=1, PADDING=0)
# a(torch.randn(1, 1, 240, 320))  # B C H W 로 가짜 데이터 넣어보고 체크


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
                f'./records/AccRun_CNN_best_model_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_channels_{model.cs}_kernel_{model.KERNEL_SIZE}_stride_{model.STRIDE}.pt')
            torch.save(
                model.state_dict(), 
                f'./records/AccRun_CNN_best_model_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_channels_{model.cs}_kernel_{model.KERNEL_SIZE}_stride_{model.STRIDE}.pth')
            valid_loss_min = valid_loss
    # save records
    pd.DataFrame(records).to_csv(
        f'./records/AccRun_CNN_training_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_channels_{model.cs}_kernel_{model.KERNEL_SIZE}_stride_{model.STRIDE}.csv', index=False)
    
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
# # Ablation: dropout 

# %%
class my_CNN_dropout(nn.Module): 

    def __init__(self, n_class, CHANNELS, KERNEL_SIZE, STRIDE, PADDING):
        # super(my_CNN, self).__init__()
        super().__init__()
        self.n_class = n_class
        self.cs = CHANNELS
        self.KERNEL_SIZE = KERNEL_SIZE
        self.STRIDE = STRIDE
        self.PADDING = PADDING

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.cs[0], kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING), 
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        ) 
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.cs[0], self.cs[1], kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING), 
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(self.cs[1], self.cs[2], kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING), 
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        
        self.fc = None

    def forward(self, x):
        
        out1 = self.layer1(x) 
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = out3.reshape(out3.size(0), -1) 
        ###
        shape1, shape2 = out3.shape[2], out3.shape[3]
        self.fc = nn.Linear(shape1 * shape2 * self.cs[2], self.n_class).to(device)
        ###
        out4 = out4.to(device)
        out = self.fc(out4)  #######################################################

        # print(x.shape)  # torch.Size([1, 1, 240, 320])
        # print(out1.shape)  # torch.Size([1, 16, 120, 160])
        # print(out2.shape)  # torch.Size([1, 32, 120, 160])
        # print(out3.shape)  # torch.Size([1, 64, 60, 80])
        # print(out4.shape) # torch.Size([1, 307200])

        return out


# MatMul shape 맞는지 체크

# a = my_CNN(n_class=63, KERNEL_SIZE=KERNEL_SIZE, STRIDE=STRIDE, PADDING=PADDING)
# a(torch.randn(1, 1, 240, 320))  # B C H W 로 가짜 데이터 넣어보고 체크


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
                f'./records/CNN_dropout_best_model_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_channels_{model.cs}_kernel_{model.KERNEL_SIZE}_stride_{model.STRIDE}.pt')
            torch.save(
                model.state_dict(), 
                f'./records/CNN_dropout_best_model_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_channels_{model.cs}_kernel_{model.KERNEL_SIZE}_stride_{model.STRIDE}.pth')
            valid_loss_min = valid_loss
    # save records
    pd.DataFrame(records).to_csv(
        f'./records/CNN_dropout_training_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_channels_{model.cs}_kernel_{model.KERNEL_SIZE}_stride_{model.STRIDE}.csv', index=False)
    
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
# # Ablation: batch norm

# %%
class my_CNN_batchnorm(nn.Module): 

    def __init__(self, n_class, CHANNELS, KERNEL_SIZE, STRIDE, PADDING):
        # super(my_CNN, self).__init__()
        super().__init__()
        self.n_class = n_class
        self.cs = CHANNELS
        self.KERNEL_SIZE = KERNEL_SIZE
        self.STRIDE = STRIDE
        self.PADDING = PADDING

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.cs[0], kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING), 
            nn.BatchNorm2d(self.cs[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        ) 
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.cs[0], self.cs[1], kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING), 
            # nn.BatchNorm2d(self.cs[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(self.cs[1], self.cs[2], kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING), 
            # nn.BatchNorm2d(self.cs[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        
        self.fc = None

    def forward(self, x):
        
        out1 = self.layer1(x) 
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = out3.reshape(out3.size(0), -1) 
        ###
        shape1, shape2 = out3.shape[2], out3.shape[3]
        self.fc = nn.Linear(shape1 * shape2 * self.cs[2], self.n_class).to(device)
        ###
        out4 = out4.to(device)
        out = self.fc(out4)  

        return out


# MatMul shape 맞는지 체크

# a = my_CNN(n_class=63, KERNEL_SIZE=KERNEL_SIZE, STRIDE=STRIDE, PADDING=PADDING)
# a(torch.randn(1, 1, 240, 320))  # B C H W 로 가짜 데이터 넣어보고 체크


# %%
def train_batchnorm(model):
    
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
                f'./records/CNN_batchnorm_best_model_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_channels_{model.cs}_kernel_{model.KERNEL_SIZE}_stride_{model.STRIDE}.pt')
            torch.save(
                model.state_dict(), 
                f'./records/CNN_batchnorm_best_model_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_channels_{model.cs}_kernel_{model.KERNEL_SIZE}_stride_{model.STRIDE}.pth')
            valid_loss_min = valid_loss
    # save records
    pd.DataFrame(records).to_csv(
        f'./records/CNN_batchnorm_training_TrainSampleSize_{len(train_loader)*train_loader.batch_size}_channels_{model.cs}_kernel_{model.KERNEL_SIZE}_stride_{model.STRIDE}.csv', index=False)
    
    return model


def test_batchnorm(model):

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
Train_epoch = 30
BATCH_SIZE = 128
LR = 5e-2

# %%
# hyperparameters

acc_runtime_dict = {}

# "optimized" hyperparameters
param_grid = {
    'CHANNELS': [[6, 6, 16]],
    'KERNEL_SIZE' : [6],                  
    'STRIDE' : [1],                       
    'PADDING' : [0]                          
}

grid_list = list(ParameterGrid(param_grid))

# %%
if __name__ == '__main__':
    
    #################### train model ########################

    sample_size_list = [0.3, 0.6, 1.0]  # [0.3, 0.6, 1.0]

    for sample_size in tqdm(sample_size_list):

        train_loader, valid_loader, test_loader = sample_and_split_dataloader(
            X_train_, X_test_, Y_train_, Y_test_, sample_size=sample_size, valid_size=0.2, batch_size=BATCH_SIZE)

        acc_runtime_dict = {}
        
        for case in tqdm(grid_list, leave=False):

            # try:
            regr = my_CNN(n_class=63, CHANNELS=case['CHANNELS'], KERNEL_SIZE=case['KERNEL_SIZE'],
                        STRIDE=case['STRIDE'], PADDING=case['PADDING']
                    ).to(device)

            start_time = datetime.now()
            model_trained = train(regr)
            train_time = datetime.now() - start_time

            acc_runtime_dict[str(case)] = dict({'train_time': train_time})
            # except:
            #     pass

        final_dict[sample_size] = acc_runtime_dict
    
    #################### save results ########################

    # save
    with open('result_AccRun_CNN.pickle', 'wb') as f:
        pickle.dump(final_dict, f, pickle.HIGHEST_PROTOCOL)

    # load
    with open('result_AccRun_CNN.pickle', 'rb') as f:
        result_RF = pickle.load(f)


    #################### final test ########################
    
    train_loader, valid_loader, test_loader = sample_and_split_dataloader(
        X_train_, X_test_, Y_train_, Y_test_, sample_size=1.0, valid_size=0.2, batch_size=128)

    best_model = torch.load(
        './records/AccRun_CNN_best_model_TrainSampleSize_12928_channels_[6, 6, 16]_kernel_6_stride_1.pt')

    start_time = datetime.now()
    test_loss = test(best_model)
    test_time = datetime.now() - start_time

    print('test_time: ', test_time.seconds)
    print('test_loss: ', test_loss)



