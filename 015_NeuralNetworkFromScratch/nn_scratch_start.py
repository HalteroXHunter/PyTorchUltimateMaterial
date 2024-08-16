
#%% packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#%% data prep
# source: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
df = pd.read_csv('heart.csv')
df.head()

#%% separate independent / dependent features
X = np.array(df.loc[ :, df.columns != 'output'])
y = np.array(df['output'])

print(f"X: {X.shape}, y: {y.shape}")

#%% Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#%% scale the data
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

#%% network class
class NeuralNetworkFromScratch:
    def __init__(self, LR, X_train, y_train, X_test, y_test) -> None:
        self.w = np.random.randn(X_train_scale.shape[1])
        self.b = np.random.randn()
        self.LR = LR
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.L_train = []
        self.L_test = []

    def activation_function(self, x):
        # sigmoid
        return 1 / (1 + np.exp(-x))
    
    def derivative_activation_function(self, x):
        # derivative of sigmoid
        return self.activation_function(x) * (1 - self.activation_function(x))
    
    def forward(self, X):
        hidden_1 = np.dot(X, self.w) + self.b
        activate_1 = self.activation_function(hidden_1)
        return activate_1
    
    def backward(self, X, y_true):
        # calc gradients
        hidden_1 = np.dot(X, self.w) + self.b
        y_pred = self.forward(X)
        dL_dpred = 2 * (y_pred - y_true)
        dpred_dhidden1 = self.derivative_activation_function(hidden_1)
        dhidden1_dw = X
        dhidden1_db = 1
        
        dL_dw = dL_dpred * dpred_dhidden1 * dhidden1_dw
        dL_db = dL_dpred * dpred_dhidden1 * dhidden1_db
        
        return dL_dw, dL_db
    
    def optimizer(self, dL_dw, dL_db):
        # update the weights
        self.w = self.w - self.LR * dL_dw
        self.b = self.b - self.LR * dL_db
    
    def train(self, epochs):
        for i in range(epochs):
            # randon position
            random_pos = np.random.randint(len(self.X_train))

            # forward pass
            y_train_true = self.y_train[random_pos]
            y_train_pred = self.forward(self.X_train[random_pos])

            # calculate the training loss
            L = (y_train_pred - y_train_true) ** 2
            self.L_train.append(L) #for checking the results

            # calculate the gradients
            dL_dw, dL_db = self.backward(self.X_train[random_pos], y_train_true)

            # update the weights
            self.optimizer(dL_dw, dL_db)

            # calculate the error for test data
            L_sum = 0
            for i in range(len(self.X_test)):
                y_test_true = self.y_test[i]
                y_test_pred = self.forward(self.X_test[i])
                L_sum += (y_test_pred - y_test_true) ** 2
            self.L_test.append(L_sum)
        return 'training done'

    
#%% Hyper parameters
LR = 0.1
ITERATIONS = 1000

#%% model instance and training
nn = NeuralNetworkFromScratch(LR=LR, X_train=X_train_scale, y_train=y_train, X_test=X_test_scale, y_test=y_test)
nn.train(ITERATIONS)
# %% check losses
sns.lineplot(x=list(range(len(nn.L_test))), y=nn.L_test)

# %% iterate over test data
total = X_test_scale.shape[0]
correct = 0
y_preds = []

for i in range(total):
    y_true = y_test[i]
    y_pred = np.round(nn.forward(X_test_scale[i]))
    y_preds.append(y_pred)
    correct += 1 if y_true == y_pred else 0

# %% Calculate Accuracy
correct / total

# %% Baseline Classifier
from collections import Counter

Counter(y_test)

# %% Confusion Matrix
confusion_matrix(y_test, y_preds)

# %%
