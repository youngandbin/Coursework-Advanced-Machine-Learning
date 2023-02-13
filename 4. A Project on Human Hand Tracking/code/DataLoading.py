# %% [markdown]
# # Save Data

# %%
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as tt
from PIL import Image
import glob
from tqdm import tqdm
# dd

# %% [markdown]
# ## label (Y)
# - Y_train (16008 rows × 63 columns)
# - Y_test (1596 rows × 63 columns)

# %%
annotation_train = pd.read_csv('data/Training/Annotation_Training.csv')
Y_train = annotation_train.iloc[3:, :]
Y_train.columns = range(Y_train.shape[1])
Y_train = Y_train.set_index(0)
Y_train.to_csv('data/Training/Y_train.csv', index=False)

annotation_test = pd.read_csv('data/Testing/Annotation_Testing.csv')
Y_test = annotation_test.iloc[3:, :]
Y_test.columns = range(Y_test.shape[1])
Y_test = Y_test.set_index(0)
Y_test.to_csv('data/Testing/Y_test.csv', index=False)

info = annotation_train.iloc[:1, :6]  # train, test 같음
info


# %% [markdown]
# ## image (X)
# - X_train
#     - depth_1_0000001.png ~ depth_1_0016008.png
# - X_test
#     - depth_1_0000001.png ~ depth_1_0001596.png

# %%
path_train = 'data/Training/depth/*.png'
path_test = 'data/Testing/depth/*.png'

transform = tt.Compose([
    tt.PILToTensor()
])


# %%
img_list = []
for filename in tqdm(glob.glob(path_train)):
    img = Image.open(filename)
    img_tensor = transform(img)  # torch.Size([1, 240, 320])
    img_list.append(img_tensor)
X_train = torch.stack(img_list)  # torch.Size([data_len, 1, 240, 320])
torch.save(X_train, 'data/Training/X_train.pt')



img_list = []
for filename in tqdm(glob.glob(path_test)):
    img = Image.open(filename)
    img_tensor = transform(img)  # torch.Size([1, 240, 320])
    img_list.append(img_tensor)
X_test = torch.stack(img_list)  # torch.Size([data_len, 1, 240, 320])
torch.save(X_test, 'data/Testing/X_test.pt')


# %%
img_list = []
for filename in tqdm(glob.glob(path_train)):
    img = Image.open(filename)
    img_tensor = transform(img)  # torch.Size([1, 240, 320])
    ###
    c_min = img_tensor.min()
    c_max = img_tensor.max()
    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
        img_tensor = img_tensor.to(torch.int8)
    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
        img_tensor = img_tensor.to(torch.int16)
    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
        img_tensor = img_tensor.to(torch.int32)
    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
        img_tensor = img_tensor.to(torch.int64)
    ###
    img_list.append(img_tensor)
X_train = torch.stack(img_list)  # torch.Size([data_len, 1, 240, 320])
torch.save(X_train, 'data/Training/X_train_diet.pt')



img_list = []
for filename in tqdm(glob.glob(path_test)):
    img = Image.open(filename)
    img_tensor = transform(img)  # torch.Size([1, 240, 320])

    ###
    c_min = img_tensor.min()
    c_max = img_tensor.max()
    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
        img_tensor = img_tensor.to(torch.int8)
    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
        img_tensor = img_tensor.to(torch.int16)
    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
        img_tensor = img_tensor.to(torch.int32)
    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
        img_tensor = img_tensor.to(torch.int64)
    ###

    img_list.append(img_tensor)
X_test = torch.stack(img_list)  # torch.Size([data_len, 1, 240, 320])
torch.save(X_test, 'data/Testing/X_test_diet.pt')


# %% [markdown]
# # permute data 
# - (윤태형) Batch, Height, Width, Channel
# - (CNN RNN 할때는 다시 permute) Batch, Channel, Height, Width 

# %%
X_train = torch.load('data/Training/X_train_diet.pt')
X_test = torch.load('data/Testing/X_test_diet.pt') 

Y_train = pd.read_csv('data/Training/Y_train.csv')
Y_test = pd.read_csv('data/Testing/Y_test.csv')

print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('Y_train: ', Y_train.shape)
print('Y_test: ', Y_test.shape)

# %%
X_train = X_train.permute([0,2,3,1])
X_train.shape

X_test = X_test.permute([0, 2, 3, 1])
X_test.shape


# %%
torch.save(X_train, 'data/Training/X_train_diet.pt')
torch.save(X_test, 'data/Testing/X_test_diet.pt')

# %%



