
###################################### predict groundwater hydrualoc properties with deep learning ###################################### 

GeoNet Module to load and preprocess data, training Pytorch deep learning regression to predict groundwater hydrualic properties for Mcihigan Lower Penensila
features: group fold cross validation based on county label


Required:
1- OneAPI AI toolkits for Pytorch
2- DataBase: HydroGeoDataset_ML.h5






# list of huc8
# 4050001 4050002 4050003 4050004 4050005 4050006 4050007 
# 4060101 4060102 4060103 4060104 4060105 
# 4070003 4070004 4070005 4070006 4070007
# 4080101 4080102 4080103 4080104 4080201 4080202 4080203 4080204 4080205 4080206
# 4090001 4090003 4090004 
# 4100001 4100002 4100013



List of numerical and categorical variables are as follows and available in global_var.py:

Numerical:
				'DEM_{RESOLUTION}m',
				
		  		'non_snow_accumulation_raster_{RESOLUTION}m', 
				'snow_accumulation_raster_{RESOLUTION}m',
				'melt_rate_raster_{RESOLUTION}m',
				'average_temperature_raster_{RESOLUTION}m',	
				'snow_layer_thickness_raster_{RESOLUTION}m',
				
				'kriging_output_AQ_THK_2_{RESOLUTION}m',
				'kriging_output_H_COND_1_{RESOLUTION}m',
				'kriging_output_AQ_THK_1_{RESOLUTION}m',
				'kriging_output_H_COND_2_{RESOLUTION}m', 
				'kriging_output_SWL_{RESOLUTION}m', 
				'kriging_output_V_COND_2_{RESOLUTION}m',
				'kriging_output_TRANSMSV_1_{RESOLUTION}m', 
				'kriging_output_TRANSMSV_2_{RESOLUTION}m', 
				'kriging_output_V_COND_1_{RESOLUTION}m', 
				
				'kriging_stderr_SWL_{RESOLUTION}m', 
   				'kriging_stderr_H_COND_1_{RESOLUTION}m',
				'kriging_stderr_H_COND_2_{RESOLUTION}m', 
				'kriging_stderr_V_COND_1_{RESOLUTION}m',
				'kriging_stderr_V_COND_2_{RESOLUTION}m', 
				'kriging_stderr_AQ_THK_1_{RESOLUTION}m', 
				'kriging_stderr_AQ_THK_2_{RESOLUTION}m',
				
				'QAMA_MILP_{RESOLUTION}m',        ## mean annual streamflow
    			'QBMA_MILP_{RESOLUTION}m',        ## Mean annual flow from excess ET
				'QCMA_MILP_{RESOLUTION}m',        ## Mean annual flow with reference gage regression
    			'QDMA_MILP_{RESOLUTION}m',        ## Mean annual flow with NHDPlusAdditionRemoval
				'QEMA_MILP_{RESOLUTION}m',        ## Mean annual flow from gage adjustment
				
    			'QIncrBMA_MILP_{RESOLUTION}m',    ## Incremental flow with excess ET
				'QIncrCMA_MILP_{RESOLUTION}m',    ## Incremental flow by subtracting upstream QCMA
				'QIncrAMA_MILP_{RESOLUTION}m',    ## 
				'QIncrDMA_MILP_{RESOLUTION}m', 	  ## Incremental flow with NHDPlusAdditionRemoval
				'QIncrEMA_MILP_{RESOLUTION}m',    ## Incremental flow from gage adjustment
				'QIncrFMA_MILP_{RESOLUTION}m',    ## Incremental flow from gage sequestration
				
				'x_{RESOLUTION}m',
				'y_{RESOLUTION}m',
				
				'VBMA_MILP_{RESOLUTION}m',      # Velocity for QBMA
				'VCMA_MILP_{RESOLUTION}m',		# Velocity for 
				'VDMA_MILP_{RESOLUTION}m',      # Velocity for QCMA
				'VEMA_MILP_{RESOLUTION}m',		# Velocity from gage adjustment


Categorical:

				'geomorphons_{RESOLUTION}m_250Dis',    
				'MI_geol_poly_{RESOLUTION}m',
				'Glacial_Landsystems_{RESOLUTION}m',
				'landuse_{RESOLUTION}m', 
				'landforms_{RESOLUTION}m_250Dis',
				'Aquifer_Characteristics_Of_Glacial_Drift_{RESOLUTION}m'


  
  
The shape of all data are the same. 

Novalues: -999

EPSG:26990



  
GeoCNN:

"""The objective of this code is to implement and train a Convolutional Neural Network (CNN) 
	integrated with a Long Short-Term Memory (LSTM) network to predict static water levels
	using both static and climate data. The code handles data preprocessing,
	including the standardization and combination of static and climate data,
	configures and trains the CNN-LSTM model, and evaluates its performance. 
	The process involves loading data, setting up the model, handling GPU resources, 
	training with gradient accumulation to manage memory usage, and saving the best performing model.
"""
