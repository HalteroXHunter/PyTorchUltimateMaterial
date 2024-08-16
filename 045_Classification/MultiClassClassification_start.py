#%% packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
import seaborn as sns
import numpy as np

# %% data import
iris = load_iris()
X = iris.data
y = iris.target

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %% convert to float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# %% dataset
class IrisDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y) 
        self.y = self.y.type(torch.LongTensor)
        self.len = self.X.shape[0]
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# %% dataloader
iris_data = IrisDataset(X_train, y_train)
train_loader = DataLoader(dataset = iris_data, batch_size=32, shuffle=True)

# %% check dims
print(f"X shape: {iris_data.X.shape}, y shape: {iris_data.y.shape}")

# %% define class
class MultiClassNet(nn.Module):
    def __init__(self, num_features, hidden, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x


# %% hyper parameters
NUM_FEATURES = X_train.shape[1] 
HIDDEN = 6
NUM_CLASSES = len(iris_data.y.unique())

# %% create model instance
model = MultiClassNet(num_features=NUM_FEATURES, hidden=HIDDEN, num_classes=NUM_CLASSES)

# %% loss function
criterion = nn.CrossEntropyLoss()
# %% optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# %% training
EPOCHS = 100
losses = []
for epoch in range(EPOCHS):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    losses.append(float(loss.data.detach().numpy()))
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
    
     
# %% show losses over epochs
print(losses)
sns.lineplot(x=range(EPOCHS), y=losses)

# %% test the model
X_test_torch = torch.from_numpy(X_test)
with torch.no_grad():
    y_test_log = model(X_test_torch)
    y_test_pred = torch.argmax(y_test_log.data, dim=1)
    print(y_test_pred)

# %% Accuracy
acc = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {acc}")
f1 = f1_score(y_test, y_test_pred, average='weighted')
print(f"F1 Score: {f1}")

# %% Naive classifier
from collections import Counter

print(Counter(y_test))
most_common = Counter(y_test).most_common()[0][1]
print(f"Naive Accuracy: {most_common/len(y_test)*100}%")


# %%
