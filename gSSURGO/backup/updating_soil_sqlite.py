import arcpy
import sqlite3
import pandas as pd
import os
import numpy as np
import pandas as pd
import numpy as np
import sqlite3

gSSURGO_base_path = r"/data/MyDataBase/SWATGenXAppData/Soil/gSSURGO_CONUS/gSSURGO_CONUS.gdb"
soil_data_base = r"C:/SWAT/SWATPlus/Databases/swatplus_soils.sqlite"
original_soil_data_base = r"D:/Downloads/swatplus_soils.sqlite"
SWAT_gSSURGO_csv_path = r"/data/MyDataBase/SWATGenXAppData/Soil/SWAT_gssurgo.csv"
arcpy.env.workspace = gSSURGO_base_path

gSSURGO_tables = ['chorizon', 'chtexturegrp', 'component']

# Function to convert ArcGIS Table to pandas DataFrame
def table_to_dataframe(table_name):
	fields = [f.name for f in arcpy.ListFields(table_name)]
	with arcpy.da.SearchCursor(table_name, fields) as cursor:
		data = list(cursor)
	return pd.DataFrame(data, columns=fields)

# Loop through the tables and save them as pickle
for table in gSSURGO_tables:
	print(f"Processing {table}")
	pickle_path = os.path.join("/data/MyDataBase/SWATGenXAppData/Soil", f"{table}.pkl")
	if os.path.exists(pickle_path):
		print(f"Skipping {table} as it already exists")
		continue
	else:
		df = table_to_dataframe(os.path.join(gSSURGO_base_path, table))
		df.to_pickle(pickle_path)
		print(f"Saved {table} to {pickle_path}")


def process_gssurgo_data():
	extracted_gssurgo_path = "/data/MyDataBase/SWATGenXAppData/Soil/gSSURGO_db.pkl"
	rewrite = True

	if not os.path.exists(extracted_gssurgo_path) or rewrite:
		writing_gSSURGO_in_pickle(extracted_gssurgo_path)
	return extracted_gssurgo_path


def writing_gSSURGO_in_pickle(extracted_gssurgo_path):

	chorizon = pd.read_pickle("/data/MyDataBase/SWATGenXAppData/Soil/chorizon.pkl")
	print("################################### 3400134:",chorizon[chorizon['mukey'] == "3400134"])
	chtexturegrp = pd.read_pickle("/data/MyDataBase/SWATGenXAppData/Soil/chtexturegrp.pkl")
	component = pd.read_pickle("/data/MyDataBase/SWATGenXAppData/Soil/component.pkl")
	chorizon_chtexturegrp = pd.merge(chorizon, chtexturegrp, on='chkey', how='left')
	gSSURGO_db = pd.merge(chorizon_chtexturegrp, component, on='cokey', how='left')

	gssurgo_to_swat_columns = {'sandtotal_r': 'sand',
							'claytotal_r': 'clay',
							'silttotal_r': 'silt',
							'awc_r': 'awc',
							'caco3_r': 'caco3',
							'ec_r': 'ec',
							'ph1to1h2o_r': 'ph',
							'dbovendry_r': 'bd',
							'ksat_r': 'soil_k',
							'albedodry_r': 'alb',
							'kwfact': 'usle_k',
							"fraggt10_r": "rock",
							'texture': 'texture',
							'hydgrp': 'hyd_grp',
							'hzdepb_r': 'dp',
							'hzdept_r': 'dp_tot',
							"comppct_r": "cmppct",
							}

	gSSURGO_db.rename(columns=gssurgo_to_swat_columns, inplace=True)

	gSSURGO_db[["hzname", 'compname', 'cokey', 'mukey', 'chkey', 'sand', 'clay', 'silt', 'awc', 'caco3', 'ec', 'ph',
				"dp", "dp_tot", 'bd', 'soil_k', 'alb', 'usle_k',
				'rock', 'texture', 'hyd_grp', ]].to_pickle(extracted_gssurgo_path)

def get_table_names(conn):
	cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
	return [row[0] for row in cursor]


def update_ssurgo_layer_table(conn, data):
	query = """
	INSERT INTO ssurgo_layer (id, soil_id, layer_num, dp, bd, awc, soil_k, carbon, clay, silt, sand, rock, alb, usle_k, ec, caco3, ph)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	"""
	conn.executemany(query, data)

def update_ssurgo_table(conn, data):
	query = """
	REPLACE INTO ssurgo (id, name, muid, seqn, s5id, cmppct, hyd_grp, dp_tot, anion_excl, perc_crk, texture)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	"""
	conn.executemany(query, data)

def process_gssurgo_data():
	extracted_gssurgo_path = "/data/MyDataBase/SWATGenXAppData/Soil/gSSURGO_db.pkl"
	rewrite = False
	if not os.path.exists(extracted_gssurgo_path) or rewrite:
		writing_gSSURGO_in_pickle(extracted_gssurgo_path)
	return extracted_gssurgo_path


def update_soil_data_base():
	extracted_gssurgo_path = process_gssurgo_data()
	gSSURGO_db = pd.read_pickle(extracted_gssurgo_path)
	print(gSSURGO_db[['mukey', 'dp_tot']].head(10))
	print(gSSURGO_db.head())
	## check if 377988 in mukey
	gSSURGO_db['s5id'] = None
	gSSURGO_db['cmppct'] = None
	gSSURGO_db['anion_excl'] = 0.5
	gSSURGO_db['perc_crk'] = 0.5
	gSSURGO_db['seqn'] = gSSURGO_db['mukey']
	gSSURGO_db['muid'] = gSSURGO_db['mukey']
	gSSURGO_db['snam'] = None
	gSSURGO_db['id'] = np.arange(1, gSSURGO_db.shape[0] + 1)
	gSSURGO_db['name'] = gSSURGO_db['compname']
	gSSURGO_db['hyd_grp'] = gSSURGO_db['hyd_grp'].str.upper()
	gSSURGO_db['texture'] = gSSURGO_db['texture'].str.upper()
	gSSURGO_db['soil_id'] = gSSURGO_db['mukey']
	gSSURGO_db['carbon'] = 0.5
	gSSURGO_db['hyd_grp'] = gSSURGO_db['hyd_grp'].fillna('')
	gSSURGO_db['texture'] = gSSURGO_db['texture'].fillna('')
	gSSURGO_db['dp'] = gSSURGO_db['dp'].fillna(0)
	gSSURGO_db['bd'] = gSSURGO_db['bd'].fillna(0)
	gSSURGO_db['awc'] = gSSURGO_db['awc'].fillna(0)
	gSSURGO_db['soil_k'] = gSSURGO_db['soil_k'].fillna(0)
	gSSURGO_db['carbon'] = gSSURGO_db['carbon'].fillna(0)
	gSSURGO_db['clay'] = gSSURGO_db['clay'].fillna(0)
	gSSURGO_db['silt'] = gSSURGO_db['silt'].fillna(0)
	gSSURGO_db['sand'] = gSSURGO_db['sand'].fillna(0)
	gSSURGO_db['rock'] = gSSURGO_db['rock'].fillna(0)
	gSSURGO_db['alb'] = gSSURGO_db['alb'].fillna(0)
	gSSURGO_db['usle_k'] = gSSURGO_db['usle_k'].fillna(0)
	gSSURGO_db['ec'] = gSSURGO_db['ec'].fillna(0)
	gSSURGO_db['caco3'] = gSSURGO_db['caco3'].fillna(0)
	### calculate depth range based on dp_tot and dp
	# Sort the data by mukey (map unit key) and top depth
	gSSURGO_db.sort_values(by=['mukey', 'dp_tot'], inplace=True)
	gSSURGO_db['layer_num'] = gSSURGO_db.groupby('cokey').cumcount() + 1
	print(gSSURGO_db[['layer_num', 'mukey', 'cokey','hzname','dp_tot','dp']].head(10))
	# Connect to the database
	conn = sqlite3.connect(soil_data_base)
	conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
	# Start a transaction
	conn.execute("BEGIN TRANSACTION")
	try:
		# Update ssurgo table
		update_ssurgo_table(conn, gSSURGO_db[['id', 'name', 'muid', 'seqn', 's5id', 'cmppct', 'hyd_grp', 'dp_tot', 'anion_excl', 'perc_crk', 'texture']].values.tolist())
		# Update ssurgo_layer table
		update_ssurgo_layer_table(conn, gSSURGO_db[['id', 'soil_id', 'layer_num', 'dp', 'bd', 'awc', 'soil_k', 'carbon', 'clay', 'silt', 'sand', 'rock', 'alb', 'usle_k', 'ec', 'caco3', 'ph']].values.tolist())
		# Commit the transaction
		conn.commit()
		print("Data updated successfully!")
	except Exception as e:
		# Rollback the transaction in case of any error
		conn.rollback()
		print("Error occurred. Transaction rolled back.")
		print(e)
	finally:
		# Close the connection
		conn.close()


update_soil_data_base()
print("Done")
