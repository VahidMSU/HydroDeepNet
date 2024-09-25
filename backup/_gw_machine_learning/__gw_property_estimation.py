##################### 3/4/2024 ##################################
import pandas as pd
import joblib
from sklearn.kernel_approximation import RBFSampler
from xgboost import XGBRFRegressor
from xgboost import XGBRegressor
from xgboost import DMatrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
from optuna.distributions import IntDistribution, CategoricalDistribution


def import_ml_database(RESOLUTION,DIC):

    base_data_path    = os.path.join(DIC, f"rasters_{RESOLUTION}m_with_observations.pk1")
    NHDPlusVAA_path =   os.path.join(DIC, "NHDPlusFlowlineVAA.pkl")
    NHDWaterbody_path = os.path.join(DIC, "NHDWaterbody.pkl")
    NHDFlowlines_path = os.path.join(DIC, "NHDFlowline.pkl")
    gSSURGO_path      = os.path.join(DIC, "SWAT_gssurgo.csv")

    
    base_data             = pd.read_pickle(base_data_path)
    VAA                   = pd.read_pickle(NHDPlusVAA_path)[['StreamLeve', 'StreamOrde',  'MinElevSmo',  'MaxElevSmo', 'Slope', "AreaSqKm", "NHDPlusID"]].rename(columns= {"AreaSqKm":"AreaSqKm_flo"})
    waterbodies           = pd.read_pickle(NHDWaterbody_path)[['Permanent_Identifier', 'AreaSqKm', 'FType', 'FCode']].rename(columns= {"AreaSqKm":"AreaSqKm_wb"})
    flowlines             = pd.read_pickle(NHDFlowlines_path)[[ 'FType', 'FCode',  'NHDPlusID', 'WBArea_Permanent_Identifier', 'Permanent_Identifier' ]]
    gSSURGO               = pd.read_csv(gSSURGO_path)[["muid", "hyd_grp", "texture"]].rename(columns= {"gSURRGO":"gSSURGO"})
    print("Data Imported")
    print(f'gSSURGO: {gSSURGO.columns}')
    flowlines_waterbodies = pd.merge(flowlines, waterbodies, left_on='WBArea_Permanent_Identifier', right_on='Permanent_Identifier', how='left', suffixes=('_flo', '_wb'))
    flowlines_waterbodies = flowlines_waterbodies[[
                                            "FType_flo",	"FCode_flo", "NHDPlusID", 
                                            "AreaSqKm_wb" , "FType_wb",	"FCode_wb"
                                            ]]

    flowlines_waterbodies_VAA = pd.merge(flowlines_waterbodies, VAA, on='NHDPlusID')

    flowlines_waterbodies_VAA_base_data = pd.merge(base_data, flowlines_waterbodies_VAA, on = 'NHDPlusID', how='right').rename(columns = {'index_left':'observed_well_data'}).rename(columns = {'gSURRGO':'gSSURGO'})

    flowlines_waterbodies_VAA_base_data_gSSURGO = flowlines_waterbodies_VAA_base_data.merge(gSSURGO, left_on = 'gSSURGO', right_on = "muid")

    return flowlines_waterbodies_VAA_base_data_gSSURGO[~flowlines_waterbodies_VAA_base_data_gSSURGO.observed_well_data.isna()].fillna(-999), flowlines_waterbodies_VAA_base_data_gSSURGO

DIC = "/home/rafieiva/MyDataBase/codes/SWAT-CONUS/gw_machine_learning/"
RESOLUTION = 250
mldata, all_data = import_ml_database(RESOLUTION, DIC)
mldata = mldata.sample(n=10000, random_state=42  )

target_variable = ['SWL']

categorical_features = [
                        "FType_wb",  "FCode_wb",
                        "FType_flo", "FCode_flo", 
                        'AQU_CHAR', 'GeoLandSy', 'GeologUnit',    
                        'landforms' ,'landuse', "geomorph",
                        "hyd_grp",    "texture",
                        'StreamLeve', 'StreamOrde', 
                        ]  

num_features = [
                'S_SWL', 'er_SWL',
                "AreaSqKm_flo", "AreaSqKm_wb" , 
                'MinElevSmo',  'MaxElevSmo',
                'Slope',
                ]  

rbf_features=['x','y','Elevation']

models= [
        (XGBRFRegressor(random_state=42, tree_method='hist', device='gpu', n_jobs=55),  
        {'model__n_estimators': IntDistribution(50, 250, step=50),
        'model__max_depth': IntDistribution(2, 10),
        'model__min_child_weight': IntDistribution(1,10),
        'model__colsample_bynode': CategoricalDistribution([0.5])}, 
        "random_forest_cpu", 'Optuna')
        ]


rbf_processor = RBFSampler(gamma=1, random_state=42)



preprocessor = ColumnTransformer(
    
    transformers = 
    [
    ('num', 'passthrough', num_features),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features),
    ('rbf', rbf_processor, rbf_features)
    ]
    )

# Nested cross-validation configuation
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

mldata = mldata[num_features+rbf_features+categorical_features+target_variable]

# Splitting the data into training and testing sets
X = mldata.drop(columns=target_variable)
y = mldata[target_variable].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def custome_scorer(estimator, X, y_true):
    y_pred = estimator.predict(X)
    return -mean_squared_error(y_true, y_pred)

for model, param_grid, model_name, search_strategy in models:
    
    pipeline = Pipeline([ 

    ('preprocessor', preprocessor),
    ('scaler', 'passthrough'), 
    ('model', model)
    ])

    pipeline.set_params(model=model) 

    optimization_search = OptunaSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        scoring=custome_scorer,
        n_trials=25,  
        cv=5,       
        n_jobs=55,
        verbose=0
    )

    # Fit the OptunaSearchCV to the training data
    optimization_search.fit(X_train, y_train)

    # Best model after optimization
    best_model = optimization_search.best_estimator_

    # Predictions on the test set
    y_pred = best_model.predict(X_test)
    # save the best model
    best_model.fit(X, y)
    # Save the best model

    model_path = os.path.join(DIC, f'{target_variable[0]}_{model_name}.joblib')
    joblib.dump(best_model, model_path)
    

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.set_title('True vs. Predicted Values')
    plt.show()
    
