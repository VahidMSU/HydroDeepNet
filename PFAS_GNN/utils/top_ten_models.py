import pandas as pd

path = 'figs_archive/results/GridSearchResults.csv'

df = pd.read_csv(path)

# Group by 'gnn_model' and get the indices of the top 10 models based on 'test_loss'
grouped = df.groupby('gnn_model').apply(lambda x: x.nsmallest(10, 'test_loss')).reset_index(drop=True)
## save
grouped.drop(columns=["sw_train_loss","sw_val_loss","sw_test_loss"]).round(5).to_csv('figs_archive/results/TopTenModels.csv', index=False)
# Print the relevant columns, including test_loss and other columns
print(grouped)
## now select the best model base don the lowest test_loss and then lowest val_loss
best_model = grouped.nsmallest(1, 'test_loss').nsmallest(1, 'val_loss')
## save the best model
best_model.drop(columns=["sw_train_loss","sw_val_loss","sw_test_loss"]).round(2).to_csv('figs_archive/results/BestModel.csv', index=False)