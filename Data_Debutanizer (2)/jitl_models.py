# Just-in-Time Learning (JITL) Module

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


class LocalNN(nn.Module):
    def __init__(self, input_dim, hidden_units=32):
        super(LocalNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class JITLModel:
    def __init__(self, k_neighbors=10, hidden_units=10, max_epochs=100, 
                 learning_rate=0.01, batch_size=None, random_state=42, device=None):
        self.k_neighbors = k_neighbors
        self.hidden_units = hidden_units
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
        
        self.X_train_db = None
        self.y_train_db = None
        self.scaler = None
        self.nn_searcher = None
        
    def fit(self, X_train, y_train):
        self.X_train_db = X_train.copy()
        self.y_train_db = y_train.copy()
        
        self.scaler = StandardScaler()
        self.X_train_db_scaled = self.scaler.fit_transform(self.X_train_db)
        
        self.nn_searcher = NearestNeighbors(n_neighbors=self.k_neighbors, 
                                            metric='euclidean', 
                                            algorithm='auto')
        self.nn_searcher.fit(self.X_train_db_scaled)
        
        print(f"JITL Training Database Created:")
        print(f"  - Training samples: {len(self.X_train_db)}")
        print(f"  - Features: {self.X_train_db.shape[1]}")
        print(f"  - k neighbors: {self.k_neighbors}")
        
    def _train_local_model(self, X_local, y_local):
        X_tensor = torch.from_numpy(X_local.astype(np.float32)).to(self.device)
        y_tensor = torch.from_numpy(y_local.astype(np.float32)).to(self.device)
        
        model = LocalNN(input_dim=X_local.shape[1], 
                       hidden_units=self.hidden_units).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        batch_size = self.batch_size if self.batch_size else len(X_local)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch_loss / len(dataloader) < 1e-6:  # early stop if converged
                break
        
        return model
    
    def predict(self, X_query):
        if self.X_train_db is None:
            raise ValueError("Model must be fitted before prediction.")
        
        X_query_scaled = self.scaler.transform(X_query)
        y_pred = np.zeros(len(X_query))
        
        print(f"\nJITL Prediction Progress:")
        print(f"  - Query points: {len(X_query)}")
        
        for i, x_query in enumerate(X_query_scaled):
            distances, indices = self.nn_searcher.kneighbors([x_query])
            
            X_local = self.X_train_db_scaled[indices[0]]
            y_local = self.y_train_db[indices[0]]
            
            local_model = self._train_local_model(X_local, y_local)
            
            local_model.eval()
            with torch.no_grad():
                x_query_tensor = torch.from_numpy(x_query.astype(np.float32)).to(self.device)
                x_query_tensor = x_query_tensor.unsqueeze(0)
                pred = local_model(x_query_tensor).cpu().numpy()[0]
            
            y_pred[i] = pred
            del local_model
            
            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(X_query)} query points...")
        
        print(f"  - Completed all {len(X_query)} predictions")
        return y_pred
    
    def fit_predict(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        y_pred_test = self.predict(X_test)
        return y_pred_test

