import os
from flask import Blueprint, request, jsonify
import h5py
def hydrogeo_dataset_dict(path=None):
	path = "/data/MyDataBase/HydroGeoDataset/HydroGeoDataset_ML_250.h5"
	with h5py.File(path,'r') as f:
		groups = f.keys()
		hydrogeo_dict = {group: list(f[group].keys()) for group in groups}
	return hydrogeo_dict

def get_subvariables(variable):
	# Fetch subvariables for the selected variable
	hydrodict = hydrogeo_dataset_dict()
	return hydrodict.get(variable, None)

if __name__ == "__main__":
	variable = "geospatial"
	print(get_subvariables(variable))