import numpy as np
import pandas as pd 

path = 'figs_archive/results/GridSearchResults.csv'

df = pd.read_csv(path).dropna()
## average train loss, val loss, test loss, test accuracy
train_loss = df['train_loss'].values    
val_loss = df['val_loss'].values
test_loss = df['test_loss'].values

print(f"Median train loss: {np.median(train_loss)}")
print(f"Median val loss: {np.median(val_loss)}")
print(f"Median test loss: {np.median(test_loss)}")