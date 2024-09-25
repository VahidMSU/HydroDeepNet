import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os
import logging
from Recharge_predictor.models import NeuralNetwork
from Recharge_predictor.data_importer import import_hru_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class HRUPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categorical_prediction = True
        self.clean_up()

    def clean_up(self):
        logging.info("Cleaning up results directory...")
        for file in os.listdir("results"):
            if file.endswith(".png") or file.endswith(".txt") or file.endswith(".pth"):
                os.remove(os.path.join("results", file))

    def preprocess_data(self, numerical_features, categorical_features, target):
        # Label encode categorical features
        label_encoders = {}
        for col in categorical_features.columns:
            le = LabelEncoder()
            categorical_features[col] = le.fit_transform(categorical_features[col])
            label_encoders[col] = le

        # Standardize numerical features
        scaler = StandardScaler()
        numerical_scaled = scaler.fit_transform(numerical_features)

        # Concatenate numerical and categorical features
        features = np.concatenate([numerical_scaled, categorical_features.values], axis=1)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target.values, test_size=0.5, random_state=42)

        # Convert to PyTorch tensors and move to GPU
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)  # Use long type for classification
        y_test = torch.tensor(y_test, dtype=torch.long).to(self.device)  # Use long type for classification

        return X_train, X_test, y_train, y_test
    
    def train_model(self, model, X_train, y_train, num_epochs=200):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(X_train) // 32, epochs=num_epochs)
        
        early_stopping_counter = 0
        early_stopping_threshold = 1e-5
        best_loss = float('inf')
        best_model_wts = model.state_dict()
        
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            logging.info(f"Epoch {epoch+1}/{num_epochs}...")
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Early Stopping Counter: {early_stopping_counter}')

            if loss.item() < best_loss - early_stopping_threshold:
                best_loss = loss.item()
                best_model_wts = model.state_dict()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter > 10:
                    logging.info("Early stopping...")
                    break

        model.load_state_dict(best_model_wts)
        return model

    def evaluate_model(self, model, X_test, y_test, file):
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            _, predicted = torch.max(y_pred, 1)
            accuracy = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
            # Save accuracy and classification report
            with open('results/accuracy.txt', 'a' if os.path.exists('results/accuracy.txt') else 'w') as f:
                f.write(f'Accuracy: {accuracy:.4f} {os.path.basename(file)}\n')
            with open('results/classification_report.txt', 'a' if os.path.exists('results/classification_report.txt') else 'w') as f:
                f.write(classification_report(y_test.cpu().numpy(), predicted.cpu().numpy()))
 
            logging.info(f'Accuracy: {accuracy:.4f}')
            logging.info(classification_report(y_test.cpu().numpy(), predicted.cpu().numpy()))
            # Save the model
            self.recharge_predictor = 'results/Recharge_predictor.pth'
            torch.save(model.state_dict(), self.recharge_predictor)

    def run(self):
        files = os.listdir("/data/MyDataBase/SWAT_ML")
        files = [file for file in files if file.endswith(".pkl")]

        for i, file in enumerate(files):
            data_path = os.path.join("/data/MyDataBase/SWAT_ML", file)
            logging.info(f"Processing file {i+1}/{len(files)}: {data_path}")
            numerical_features, categorical_features, target = import_hru_data(data_path)
            logging.info(f"Numerical features: {numerical_features.shape}, Categorical features: {categorical_features.shape}, Target: {target.shape}")

            X_train, X_test, y_train, y_test = self.preprocess_data(numerical_features, categorical_features, target)
            input_dim = X_train.shape[1]
            logging.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
            if i == 0:
                num_classes = len(np.unique(target))
                model = NeuralNetwork(input_dim, num_classes).to(self.device)
            elif model.fc_initial.in_features != input_dim:
                logging.info(f"Model input dimension: {model.fc_initial.in_features}, Data input dimension: {input_dim}")
                logging.info("Input dimension mismatch detected. continue.")
                continue
            logging.info(f"Model input dimension: {model.fc_initial.in_features}, Data input dimension: {input_dim}")
            model = self.train_model(model, X_train, y_train)

            self.evaluate_model(model, X_test, y_test, file)

if __name__ == "__main__":
    hru_predictor = HRUPredictor("/data/MyDataBase/SWAT_ML")
    hru_predictor.run()
