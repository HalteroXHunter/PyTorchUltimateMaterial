#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

#%% model class
class LinearRegressionTor(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionTor, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)

input_dim = 1
output_dim = 1
model = LinearRegressionTor(input_dim, output_dim)

#%% loss and optimizer
loss_fun = nn.MSELoss()

LR = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

#%% training
losses, slope, bias = [], [], []

num_epochs = 10000

for epoch in range(num_epochs):
    # set gradients to zero
    optimizer.zero_grad()

    # forward pass
    y_pred = model(X)
    loss = loss_fun(y_pred, y_true)
    
    # backward pass
    loss.backward()
    # update weights
    optimizer.step()

    # get parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])
    # store losses
    losses.append(float(loss.data))
        
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, loss: {loss.data}')
# %% visualize model training
# BASELINE: bias:37.285, slope:-5.344
sns.scatterplot(x=range(num_epochs), y=losses)


# %% visualize bias terms
sns.scatterplot(x=range(num_epochs), y=bias)

# %% visualize slope terms
sns.scatterplot(x=range(num_epochs), y=slope)

# %% check the result
y_pred = model(X).data.numpy().reshape(-1)
sns.scatterplot(x=X_list, y=y_list)
sns.lineplot(x=X_list, y=y_pred, color='red')

# %% check the result
model.eval()
y_pred = [i[0] for i in model(X).data.numpy()]
y = [i[0] for i in y_true.data.numpy()]
sns.scatterplot(x=X_list, y=y)
sns.lineplot(x=X_list, y=y_pred, color='red')
# %%
