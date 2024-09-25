
def huc8_list_loader():

	list_of_regions = [

				'4050001', '4050002', '4050003', '4050004', 
				'4050005', '4050006', '4050007', '4060101',
				'4060102', '4060103', '4060104', '4060105', 
				'4070003', '4070004', '4070005', '4070006', 
				'4070007', '4080205', '4100013', '4090004',
				'4080101', '4080102', '4080103', '4080104',
				'4080201', '4080202', '4080203', '4080204', 
				'4080206', '4090001', '4090003', '4100001',
				'4100002', 

				]
	
	return list_of_regions


def county_list_loader():
	list_of_counties = [
		'1', '3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15',
		'16', '18', '19', '20', '23', '24', '25', '26', '28', '29', '30', '32', '33', '34',
		'35', '37', '38', '39', '40', '41', '43', '44', '45', '46', '47', '50', '51', '53',
		'54', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '67', '68', '69',
		'70', '71', '72', '73', '74', '75', '76', '78', '79', '80', '81', '82', '83'
	]

	return list_of_counties


def get_var_name(feature_type, RESOLUTION):
	
	if feature_type == 'numerical':
		
		return [

#				f'lon_{RESOLUTION}m',
#				f'lat_{RESOLUTION}m',
#				f'x_{RESOLUTION}m',
#				f'y_{RESOLUTION}m',
			#	f'DEM_{RESOLUTION}m',
		    #    f'obs_SWL_{RESOLUTION}m',

		  		f'non_snow_accumulation_raster_{RESOLUTION}m', 
				f'snow_accumulation_raster_{RESOLUTION}m',
				f'melt_rate_raster_{RESOLUTION}m',
				f'average_temperature_raster_{RESOLUTION}m',	
			
				f'snow_layer_thickness_raster_{RESOLUTION}m',

			#	f"ppt_2010_{RESOLUTION}m",
			#	f"ppt_2017_250m",
			#	f"ppt_2018_250m",
			#	f'kriging_output_H_COND_1_{RESOLUTION}m',
			#	f'kriging_output_SWL_{RESOLUTION}m',
			#	f'kriging_output_AQ_THK_1_{RESOLUTION}m',
	#			f'kriging_output_H_COND_2_{RESOLUTION}m', 
#				f'kriging_output_V_COND_2_{RESOLUTION}m',
#				f'kriging_output_TRANSMSV_1_{RESOLUTION}m', 
#				f'kriging_output_TRANSMSV_2_{RESOLUTION}m', 
			#	f'kriging_output_V_COND_1_{RESOLUTION}m', 

			#	f'kriging_output_{target_array.split("obs_")[1]}',
			#	f'kriging_stderr_{target_array.split("obs_")[1]}',

			#	f'kriging_stderr_SWL_{RESOLUTION}m', 
    
   			#	f'kriging_stderr_H_COND_1_{RESOLUTION}m',
			#	f'kriging_stderr_H_COND_2_{RESOLUTION}m', 
			#	f'kriging_stderr_V_COND_1_{RESOLUTION}m',
			#	f'kriging_stderr_V_COND_2_{RESOLUTION}m', 
			#	f'kriging_stderr_AQ_THK_1_{RESOLUTION}m', 
			#	'kriging_stderr_AQ_THK_2_{RESOLUTION}m',
    
			#	f'QAMA_MILP_{RESOLUTION}m',        ## mean annual streamflow
    		#	f'QBMA_MILP_{RESOLUTION}m',        ## Mean annual flow from excess ET
			#	f'QCMA_MILP_{RESOLUTION}m',        ## Mean annual flow with reference gage regression
    		#	f'QDMA_MILP_{RESOLUTION}m',        ## Mean annual flow with NHDPlusAdditionRemoval
			#	f'QEMA_MILP_{RESOLUTION}m',        ## Mean annual flow from gage adjustment
    		
			#	f'QIncrBMA_MILP_{RESOLUTION}m',
			#	'QIncrCMA_MILP_{RESOLUTION}m', 
			#	'QFMA_MILP_{RESOLUTION}m',
    
			#	f'QGAdjMA_MILP_{RESOLUTION}m', 
			#	f'QIncrAMA_MILP_{RESOLUTION}m',
			#	'QIncrDMA_MILP_{RESOLUTION}m', 
			#	'QIncrEMA_MILP_{RESOLUTION}m', 
			#	'QIncrFMA_MILP_{RESOLUTION}m',

			#	f'VBMA_MILP_{RESOLUTION}m',
			#	'VCMA_MILP_{RESOLUTION}m',
			#	'VDMA_MILP_{RESOLUTION}m',
			#	'VEMA_MILP_{RESOLUTION}m',
			## population density
		#	f"pden1990_ML_{RESOLUTION}m",
		#	f"pden2000_ML_{RESOLUTION}m",
			f"pden2010_ML_{RESOLUTION}m",

			f"LC20_Asp_220_{RESOLUTION}m",
			f"LC20_BPS_220_{RESOLUTION}m",
			f"LC20_EVT_220_{RESOLUTION}m",
			f"LC20_Elev_220_{RESOLUTION}m",
			f"LC20_SlpD_220_{RESOLUTION}m",
			f"LC20_SlpP_220_{RESOLUTION}m",
			f"LC22_EVC_220_{RESOLUTION}m",
			f"LC22_EVH_220_{RESOLUTION}m",
	#gssurgo:
	#	'soil_k', 'dp', 'bd', 'awc', 'carbon',
     #  'clay', 'silt', 'sand', 'rock', 'alb', 'usle_k', 'ec', 'caco3', 'ph',
     #  'dp_tot'
			f'gssurgo/soil_k',
			f'gssurgo/dp',
			f'gssurgo/bd',
			f'gssurgo/awc',
			f'gssurgo/carbon',
			f'gssurgo/clay',
			f'gssurgo/silt',
			f'gssurgo/sand',
			f'gssurgo/rock',
#			f'gssurgo/alb',
#			f'gssurgo/usle_k',
#			f'gssurgo/ec',
#			f'gssurgo/caco3',
#			f'gssurgo/ph',
	#		f'gssurgo/dp_tot',
			

			]	
	
	if feature_type == 'categorical':

		return [

			#	f'geomorphons_{RESOLUTION}m_250Dis', 
			#	f'MI_geol_poly_{RESOLUTION}m',
			#	f'Glacial_Landsystems_{RESOLUTION}m',
			#	f'COUNTY_{RESOLUTION}m',
			#    f'gSURRGO_swat_{RESOLUTION}m',
				f'landuse_{RESOLUTION}m', 
			#	f'landforms_{RESOLUTION}m_250Dis',
			#	f'Aquifer_Characteristics_Of_Glacial_Drift_{RESOLUTION}m'

				]
