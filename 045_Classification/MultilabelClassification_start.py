#%% packages
from ast import Mult
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import seaborn as sns
import numpy as np
from collections import Counter
# %% data prep
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size = 0.2)


# %% dataset and dataloader
class MultilabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# TODO: create instance of dataset
multilabel_data = MultilabelDataset(X_train, y_train)
multilabel_test_data = MultilabelDataset(X_test, y_test)

# TODO: create train loader
train_loader = DataLoader(dataset=multilabel_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=multilabel_test_data, batch_size=32, shuffle=True)

print(f"X shape: {multilabel_data.X.shape}, y shape: {multilabel_data.y.shape}")

# %% model
# TODO: set up model class

class MultiLabelNet(nn.Module):
    def __init__(self, num_features, hidden, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# TODO: define input and output dim
NUM_FEATURES = multilabel_data.X.shape[1]
HIDDEN = 20
NUM_CLASSES = multilabel_data.y.shape[1]

# TODO: create a model instance
model = MultiLabelNet(NUM_FEATURES, HIDDEN, NUM_CLASSES)
model.train()

# %% loss function, optimizer, training loop
# TODO: set up loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
slope, bias = [], []
number_epochs = 100

# TODO: implement training loop
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the GPU
model.to(device)

for epoch in range(number_epochs):
    for j, (X, y) in enumerate(train_loader):

        # Move data to the GPU
        inputs, labels = X.to(device), y.to(device)
        
        # optimization
        optimizer.zero_grad()

        # forward pass
        y_hat = model(inputs)

        # compute loss
        loss = loss_fn(y_hat, labels)
                
        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

    if (epoch % 10 == 0):
        print(f"Epoch {epoch}, Loss: {loss.data}")
        losses.append(loss.item())
        
    # TODO: print epoch and loss at end of every 10th epoch
    
    
# %% losses
sns.scatterplot(x=range(len(losses)), y=losses, alpha=0.5)

# %% test the model
X_test_torch = torch.FloatTensor(X_test).to(device)
with torch.no_grad():
    y_test_hat = model(X_test_torch).round()

#%% Naive classifier accuracy
# TODO: convert y_test tensor [1, 1, 0] to list of strings '[1. 1. 0.]'
y_test_str = [str(i) for i in y_test.detach().numpy()]
print(y_test_str)

# TODO: get most common class count
most_common_cnt = Counter(y_test_str).most_common()[0][1]
print(f"Naive classifier: {most_common_cnt/len(y_test_str) * 100}%")

# %% Test accuracy
# TODO: get test set accuracy
acc = accuracy_score(y_test.cpu().numpy(), y_test_hat.cpu().numpy())
print(f"Test accuracy: {acc * 100}%")

# %%
