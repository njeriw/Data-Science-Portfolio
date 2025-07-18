import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

# Load preprocessed data
train_df = pd.read_csv('/Users/njeriwhite/projects/ann/venv/train.csv')
test_df = pd.read_csv('/Users/njeriwhite/projects/ann/venv/test.csv')
val_df = pd.read_csv('/Users/njeriwhite/projects/ann/venv/val.csv') 

# View the first 5 rows of training set
train_df.head()

X_train = train_df.drop('sus_label', axis=1)
y_train = train_df['sus_label']
X_test = test_df.drop('sus_label', axis=1)
y_test = test_df['sus_label']
X_val = val_df.drop('sus_label', axis=1)
y_val = val_df['sus_label']

train_columns = X_train.select_dtypes(include=np.number).columns
numerical_transformer = StandardScaler()


preprocessor = ColumnTransformer(
    transformers=[
        ('num',numerical_transformer, train_columns),
    ]
)

pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor)
    ]
)

X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)
X_val_processed = pipeline.transform(X_val)

X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1,1)

class NeuralNetwork(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_features,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,1),
            nn.Sigmoid(),
        )
        self._initialize_weights()

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def _initialize_weights(self):
        for m in self.modules(): 
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

input_features = X_train_tensor.shape[1]
model = NeuralNetwork(input_features)

l2_regularization_strength = 1e-5
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=l2_regularization_strength)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

def train_loop(train_loader, model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        if (epoch+1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}')

epochs = 10
train_loop(train_loader, model, criterion, optimizer, epochs)

def test_loop(model, X_test_tensor, X_val_tensor): 
    model.eval()
    with torch.no_grad():      
        y_pred_tensor = model(X_test_tensor) 
        y_pred_np = (y_pred_tensor > 0.5).float().cpu().numpy()
        
        y_val_pred_tensor = model(X_val_tensor) 
        y_val_pred_np = (y_val_pred_tensor > 0.5).float().cpu().numpy()

    return y_pred_np, y_val_pred_np


y_pred_for_accuracy, y_val_pred_for_accuracy = test_loop(model, X_test_tensor, X_val_tensor)


y_test_np = y_test_tensor.cpu().numpy()
y_val_np = y_val_tensor.cpu().numpy()


test_accuracy = accuracy_score(y_pred_for_accuracy, y_test_np)
val_accuracy = accuracy_score(y_val_pred_for_accuracy, y_val_np)

print("Validation accuracy: {0}".format(val_accuracy))
print("Testing accuracy: {0}".format(test_accuracy))