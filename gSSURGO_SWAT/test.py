import geopandas as gpd
import os
import pandas as pd
import sqlite3
swat_gssurgo_path = "D:/Downloads/swatplus_soils.sqlite" #"C:/SWAT/SWATPlus/Databases/swatplus_soils.sqlite"
customized_swat_gssurgo_path = "C:/SWAT/SWATPlus/Databases/swatplus_soils.sqlite"

def read_and_save_ssurgo_data(swat_gssurgo_path):
    ### save the dataframe in csv
    conn = sqlite3.connect(swat_gssurgo_path)
    sql = "SELECT * FROM ssurgo"
    data = pd.read_sql(sql, conn)
    data.to_csv('ssurgo_data_original.csv')

    ### also ssurgo_layer
    sql = "SELECT * FROM ssurgo_layer"
    data = pd.read_sql(sql, conn)
    data.to_csv('ssurgo_layer_data_original.csv')

    conn.close()
    return data
## print the data schema of the sqlite database
read_and_save_ssurgo_data(swat_gssurgo_path)
import os
import sqlite3

swat_gssurgo_path = "D:/Downloads/swatplus_soils.sqlite"  # "C:/SWAT/SWATPlus/Databases/swatplus_soils.sqlite"
customized_swat_gssurgo_path = "C:/SWAT/SWATPlus/Databases/swatplus_soils.sqlite"

def compare_soil_data(path1, path2, muid):
    assert os.path.exists(path1), f"Path does not exist: {path1}"
    assert os.path.exists(path2), f"Path does not exist: {path2}"

    conn1 = sqlite3.connect(path1)
    cursor1 = conn1.cursor()
    conn2 = sqlite3.connect(path2)
    cursor2 = conn2.cursor()

    cursor1.execute("SELECT * FROM ssurgo WHERE muid = ?", (muid,))
    cursor2.execute("SELECT * FROM ssurgo WHERE muid = ?", (muid,))
    data1 = cursor1.fetchall()
    data2 = cursor2.fetchall()

    print(f"Data from {path1} (ssurgo table):")
    print(data1)
    print(f"Data from {path2} (ssurgo table):")
    print(data2)

    cursor1.execute("SELECT * FROM ssurgo_layer WHERE soil_id IN (SELECT seqn FROM ssurgo WHERE muid = ?)", (muid,))
    cursor2.execute("SELECT * FROM ssurgo_layer WHERE soil_id IN (SELECT seqn FROM ssurgo WHERE muid = ?)", (muid,))
    layer_data1 = cursor1.fetchall()
    layer_data2 = cursor2.fetchall()

    print(f"Data from {path1} (ssurgo_layer table):")
    for layer in layer_data1:
        print(layer)

    print(f"Data from {path2} (ssurgo_layer table):")
    for layer in layer_data2:
        print(layer)

    conn1.close()
    conn2.close()

# Example usage
muid = '377988'
compare_soil_data(swat_gssurgo_path, customized_swat_gssurgo_path, muid)



def print_schema(path):
	assert os.path.exists(path), f"Path does not exist: {path}"
	conn = sqlite3.connect(path)
	cursor = conn.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
	tables = cursor.fetchall()
	for table_name in tables:
		table_name = table_name[0]
		print(f"Table: {table_name}")
		cursor.execute(f"PRAGMA table_info({table_name})")
		columns = cursor.fetchall()
		for column in columns:
			print(column)
		print("\n")
	conn.close()

def check(path):
	assert os.path.exists(path), f"Path does not exist: {path}"
	### check 377988 in muid
	conn = sqlite3.connect(path)
	cursor = conn.cursor()
	cursor.execute("SELECT muid FROM ssurgo")
	muids = cursor.fetchall()
	muids = [muid[0] for muid in muids]
	if "377988" in muids:
		print("377988 is in the muid")
	else:
		print("377988 is not in the muid")
	conn.close()
#check(customized_swat_gssurgo_path)
#print_schema(swat_gssurgo_path)
