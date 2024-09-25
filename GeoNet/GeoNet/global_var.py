def get_var_name(feature_type, target_array, RESOLUTION, YEAR):
	if feature_type == 'numerical':
		return [
         		f'DEM_{RESOLUTION}m',
				f'x_{RESOLUTION}m',
				f'y_{RESOLUTION}m',
		 # 		f'non_snow_accumulation_raster_{RESOLUTION}m', 
		#		f'snow_accumulation_raster_{RESOLUTION}m',
		#		f'melt_rate_raster_{RESOLUTION}m',
		#		f'average_temperature_raster_{RESOLUTION}m',	
			#	f'snow_layer_thickness_raster_{RESOLUTION}m',
			#	f'snowpack_sublimation_rate_raster_{RESOLUTION}m',
				f'kriging_output_{target_array.split("obs_")[1]}',
				f'kriging_stderr_{target_array.split("obs_")[1]}',
			#	f"ppt_{YEAR}_{RESOLUTION}m",

			
			#	'kriging_output_H_COND_1_{RESOLUTION}m',
			#	'kriging_output_AQ_THK_1_{RESOLUTION}m',
			#	'kriging_output_H_COND_2_{RESOLUTION}m', 
	
 			#	f'kriging_output_SWL_{RESOLUTION}m', 
    
			#	'kriging_output_V_COND_2_{RESOLUTION}m',
			#	'kriging_output_TRANSMSV_1_{RESOLUTION}m', 
			#	'kriging_output_TRANSMSV_2_{RESOLUTION}m', 
			#	'kriging_output_V_COND_1_{RESOLUTION}m', 
			#	'kriging_stderr_SWL_{RESOLUTION}m', 
    
   			#	'kriging_stderr_H_COND_1_{RESOLUTION}m',
			#	'kriging_stderr_H_COND_2_{RESOLUTION}m', 
			#	'kriging_stderr_V_COND_1_{RESOLUTION}m',
			#	'kriging_stderr_V_COND_2_{RESOLUTION}m', 
			#	'kriging_stderr_AQ_THK_1_{RESOLUTION}m', 
			#	'kriging_stderr_AQ_THK_2_{RESOLUTION}m',
    
			#	f'QAMA_MILP_{RESOLUTION}m',
    		#	f'QBMA_MILP_{RESOLUTION}m', 
			#	'QCMA_MILP_{RESOLUTION}m', 
    		#	'QDMA_MILP_{RESOLUTION}m', 
			#	'QEMA_MILP_{RESOLUTION}m',  
    
    		#	f'QIncrBMA_MILP_{RESOLUTION}m',
			#	f'QIncrCMA_MILP_{RESOLUTION}m', 
			#	f'QFMA_MILP_{RESOLUTION}m',
    
			#	f'QGAdjMA_MILP_{RESOLUTION}m', 
			#	f'QIncrAMA_MILP_{RESOLUTION}m',
			#	f'QIncrDMA_MILP_{RESOLUTION}m', 
			#	'QIncrEMA_MILP_{RESOLUTION}m', 
			#	'QIncrFMA_MILP_{RESOLUTION}m',

			#	f'VBMA_MILP_{RESOLUTION}m',
			#	f'VCMA_MILP_{RESOLUTION}m',
			#	'VDMA_MILP_{RESOLUTION}m',
			#	'VEMA_MILP_{RESOLUTION}m',



			## population density
		#	f"pden1990_ML_{RESOLUTION}m",
		#	f"pden2000_ML_{RESOLUTION}m",
		#	f"pden2010_ML_{RESOLUTION}m",


		#	f"LC20_Asp_220_{RESOLUTION}m",
		#	f"LC20_BPS_220_{RESOLUTION}m",
		#	f"LC20_EVT_220_{RESOLUTION}m",
		#	f"LC20_Elev_220_{RESOLUTION}m",
		#	f"LC20_SlpD_220_{RESOLUTION}m",
		#	f"LC20_SlpP_220_{RESOLUTION}m",
		#	f"LC22_EVC_220_{RESOLUTION}m",
		#	f"LC22_EVH_220_{RESOLUTION}m",
			]	
	
	if feature_type == 'categorical':

		return [
		#		f"HUC12_{RESOLUTION}m",
		#		f"NHDPlusID_{RESOLUTION}m",
				f"gSURRGO_swat_{RESOLUTION}m",
		#		f"GageAdjMA_MILP_{RESOLUTION}m",
		#		f'geomorphons_{RESOLUTION}m_250Dis', 
				f'MI_geol_poly_{RESOLUTION}m',
				f'Glacial_Landsystems_{RESOLUTION}m',
		#		f'COUNTY_{RESOLUTION}m',
		#		f'landuse_{RESOLUTION}m', 
		#		f'landforms_{RESOLUTION}m_250Dis',
				f'Aquifer_Characteristics_Of_Glacial_Drift_{RESOLUTION}m'

				]
