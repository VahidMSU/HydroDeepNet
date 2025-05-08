import sqlite3
import pandas as pd

path = "/usr/local/share/SWATPlus/Databases/swatplus_soils.sqlite"

conn = sqlite3.connect(path)

cursor = conn.cursor()

## layers

print(cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall())

## save ssurgo_layer and ssurgo as csv files

# Get column names from the table schema
cursor.execute("PRAGMA table_info(ssurgo_layer)")
ssurgo_layer_columns = [row[1] for row in cursor.fetchall()]

cursor.execute("PRAGMA table_info(ssurgo)")
ssurgo_columns = [row[1] for row in cursor.fetchall()]

ssurgo_layer = cursor.execute("SELECT * FROM ssurgo_layer;").fetchall()
ssurgo = cursor.execute("SELECT * FROM ssurgo;").fetchall()

print("\nSSURGO Layer columns:", ssurgo_layer_columns)
print("Number of columns in data:", len(ssurgo_layer[0]) if ssurgo_layer else 0)

ssurgo_layer_df = pd.DataFrame(ssurgo_layer, columns=ssurgo_layer_columns)
ssurgo_df = pd.DataFrame(ssurgo, columns=ssurgo_columns)

ssurgo_layer_df.to_csv("ssurgo_layer.csv", index=False)
ssurgo_df.to_csv("ssurgo.csv", index=False)
