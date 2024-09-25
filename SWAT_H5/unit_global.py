### the purpose of this code is to provide a mechanism to return SWAT+ parameter unit

channel_units = {
	'jday': 'day',
	'mon': 'month',
	'day': 'day',
	'yr': 'year',
	'unit': 'unit',
	'gis_id': 'unit',
	'name': 'unit',
	'area': 'ha',
	'precip': 'm^3',
	'evap': 'm^3',
	'seep': 'm^3',
	'flo_stor': 'm^3',
	'sed_stor': 'tons',
	'orgn_stor': 'kgN',
	'sedp_stor': 'kgP',
	'no3_stor': 'kgN',
	'solp_stor': 'kgP',
	'chla_stor': 'kg',
	'nh3_stor': 'kgN',
	'no2_stor': 'kgN',
	'cbod_stor': 'kg',
	'dox_stor': 'tons',
	'san_stor': 'tons',
	'sil_stor': 'tons',
	'cla_stor': 'tons',
	'sag_stor': 'tons',
	'lag_stor': 'tons',
	'grv_stor': 'tons',
	'flo_in': 'm^3/s',
	'sed_in': 'tons',
	'orgn_in': 'kgN',
	'sedp_in': 'kgP',
	'no3_in': 'kgN',
	'solp_in': 'kgP',
	'nh3_in': 'kgN',
	'no2_in': 'kgN',
	'cbod_in': 'kg',
	'dox_in': 'tons',
	'san_in': 'tons',
	'sil_in': 'tons',
	'cla_in': 'tons',
	'sag_in': 'tons',
	'lag_in': 'tons',
	'grv_in': 'tons',
	'flo_out': 'm^3/s',
	'sed_out': 'tons',
	'orgn_out': 'kgN',
	'sedp_out': 'kgP',
	'no3_out': 'kgN',
	'solp_out': 'kgP',
	'chla_out': 'kg',
	'nh3_out': 'kgN',
	'no2_out': 'kgN',
	'cbod_out': 'kg',
	'dox_out': 'tons',
	'san_out': 'tons',
	'sil_out': 'tons',
	'cla_out': 'tons',
	'sag_out': 'tons',
	'lag_out': 'tons',
	'grv_out': 'tons',
	'water_temp': 'degc',
	'chla_in': 'kg',
}

hrus_units = {
	'precip': 'mm',
	'snofall': 'mm',
	'snomlt': 'mm',
	'surq_gen': 'mm',
	'latq': 'mm',
	'wateryld': 'mm',
	'perc': 'mm',
	'et': 'mm',
	'ecanopy': 'mm',
	'esoil': 'mm',
	'surq_cont': 'mm',
	'cn': '---',
	'sw_init': 'mm',
	'sw_final': 'mm',
	'sw_ave': 'mm',
	'sw_300': 'mm',
	'sno_init': 'mm',
	'sno_final': 'mm',
	'snopack': 'mm',
	'pet': 'mm',
	'qtile': 'mm',
	'irr': 'mm',
	'surq_runon': 'mm',
	'latq_runon': 'mm',
	'overbank': 'mm',
	'surq_cha': 'mm',
	'surq_res': 'mm',
	'surq_ls': 'mm',
	'latq_cha': 'mm',
	'latq_res': 'mm',
	'latq_ls': 'mm',
	'gwtranq': 'mm',
	'satex': 'mm',
	'satex_chan': 'mm',
	'sw_change': 'mm',
	'lagsurf': 'mm',
	'laglatq': 'mm',
	'lagsatex': 'mm',
	'wet_out': 'mm',
}

def get_channel_unit(parameter):
	"""
	:param parameter: str, the parameter name
	:return: str, the unit of the parameter
	"""
	return channel_units.get(parameter, 'None')
	
def list_of_variables(dataset_name):
	if dataset_name == 'hru_wb':
		return ['perc', 'et', 'pet', 'snomlt', 
	'surq_cha', 'latq', 'wateryld', 'perc', 'et',
	'ecanopy', 'eplant', 'esoil', 'surq_cont', 
	'cn', 'sw_init', 'sw_final', 'sw_ave', 
	'sw_300', 'sno_init', 'sno_final', 'snopack',
	'pet', 'qtile', 'irr', 'surq_runon',
	'latq_runon', 'overbank', 'surq_cha', 
	'surq_res', 'surq_ls', 'latq_cha', 
	'latq_res', 'latq_ls', 'gwtranq', 
	'satex', 'satex_chan', 'sw_change',
		'lagsurf', 'laglatq', 'lagsatex', 
		'wet_out', 'snopack', 'esoil', 
		'latq', 'precip', 'snofall']
	
	if dataset_name == 'channel_sd':		
		return ['precip', 'evap', 'seep', 'flo_stor', 'sed_stor',
		'orgn_stor', 'sedp_stor', 'no3_stor', 'solp_stor', 
		'chla_stor', 'nh3_stor', 'no2_stor', 'cbod_stor', 
		'dox_stor', 'san_stor', 'sil_stor', 'cla_stor', 
		'sag_stor', 'lag_stor', 'grv_stor', 
		'flo_in', 'sed_in', 'orgn_in', 'sedp_in', 
		'no3_in', 'solp_in', 'chla_in', 'nh3_in', 
		'no2_in', 'cbod_in', 'dox_in', 'san_in',
		'sil_in', 'cla_in', 'sag_in', 'lag_in',
		'grv_in', 'flo_out', 'sed_out', 
		'orgn_out', 'sedp_out', 'no3_out', 'solp_out', 
		'chla_out', 'nh3_out', 'no2_out', 'cbod_out', 
		'dox_out', 'san_out', 'sil_out', 'cla_out',
		'sag_out', 'lag_out', 'grv_out', 
		'water_temp']
def get_hrus_unit(parameter):
	"""
	:param parameter: str, the parameter name
	:return: str, the unit of the parameter
	"""
	return hrus_units.get(parameter, 'None')


if __name__ == '__main__':
	print(get_channel_unit('precip'))
	print(get_hrus_unit('precip'))