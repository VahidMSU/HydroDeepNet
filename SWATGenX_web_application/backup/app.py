import os
import shutil
import time
from flask import Flask, request, render_template, send_file, jsonify
from waitress import serve
import sys
sys.path.append(r'/data/MyDataBase/SWATGenXAppData/codes/NHDPlus_SWAT')
sys.path.append(r'/data/MyDataBase/SWATGenXAppData/codes/ModelProcessing')
from NHDPlus_SWAT.SWATGenXCommand import SWATGenXCommand
from NHDPlus_SWAT.integrate_streamflow_data import integrate_streamflow_data
from ModelProcessing.core import process_SCV_SWATGenXModel

app = Flask(__name__)

def single_model_creation(site_no, ls_resolution, dem_resolution, calibration_flag, validation_flag, sensitivity_flag, cal_pool_size, sen_pool_size, sen_total_evaluations, num_levels, max_cal_iterations, verification_samples):
    # Print all received parameters
    print(f"site_no: {site_no}")
    print(f"ls_resolution: {ls_resolution}")
    print(f"dem_resolution: {dem_resolution}")
    print(f"calibration_flag: {calibration_flag}")
    print(f"validation_flag: {validation_flag}")
    print(f"sensitivity_flag: {sensitivity_flag}")
    print(f"cal_pool_size: {cal_pool_size}")
    print(f"sen_pool_size: {sen_pool_size}")
    print(f"sen_total_evaluations: {sen_total_evaluations}")
    print(f"num_levels: {num_levels}")
    print(f"max_cal_iterations: {max_cal_iterations}")
    print(f"verification_samples: {verification_samples}")

    BASE_PATH = r'/data/MyDataBase/SWATGenXAppData/'
    LEVEL = "huc12"
    MAX_AREA = 500
    MIN_AREA = 100
    GAP_percent = 10

    landuse_product = "NLCD"
    landuse_epoch = "2021"

    station_name = site_no
    functionality = True
    cal_functionality = True

    try:
        if functionality:
            model_path = SWATGenXCommand(BASE_PATH, LEVEL, MAX_AREA, MIN_AREA, GAP_percent, landuse_product, landuse_epoch, ls_resolution, dem_resolution, station_name, single_model=True, random_model_selection=False, multiple_model_creation=False, target_VPUID=None)
        if cal_functionality:
            process_SCV_SWATGenXModel(NAME=station_name, sensitivity_flag=sensitivity_flag, calibration_flag=calibration_flag, verification_flag=validation_flag, START_YEAR=2015, END_YEAR=2022, nyskip=3, sen_total_evaluations=sen_total_evaluations, sen_pool_size=sen_pool_size, num_levels=num_levels, cal_pool_size=cal_pool_size, max_cal_iterations=max_cal_iterations, termination_tolerance=10, epsilon=0.0001, Ver_START_YEAR=2004, Ver_END_YEAR=2022, Ver_nyskip=3, range_reduction_flag=False, pet=2, cn=1, no_value=1e6, verification_samples=25)

        output_path = os.path.join("D:/Generated_models", f"{site_no}")
        os.makedirs("D:/Generated_models", exist_ok=True)
        shutil.make_archive(output_path, 'zip', model_path)
        return f"{output_path}.zip"

    except Exception as e:
        print(e)
        return "Failed to create SWAT+ model"

@app.route('/', methods=['GET', 'POST'])
def home():
    output = None
    if request.method == 'POST':
        site_no = request.form['user_input']
        try:
            ls_resolution = min(int(request.form.get('ls_resolution', 250)), 500)
            dem_resolution = min(int(request.form.get('dem_resolution', 30)), 250)
            calibration_flag = request.form.get('calibration_flag', False)
            validation_flag = request.form.get('validation_flag', False)
            sensitivity_flag = request.form.get('sensitivity_flag', False)
            cal_pool_size = min(int(request.form.get('cal_pool_size', 50)), 100)
            sen_pool_size = min(int(request.form.get('sen_pool_size', 180)), 500)
            sen_total_evaluations = min(int(request.form.get('sen_total_evaluations', 1000)), 5000)
            num_levels = min(int(request.form.get('num_levels', 10)), 20)
            max_cal_iterations = min(int(request.form.get('max_cal_iterations', 25)), 50)
            verification_samples = min(int(request.form.get('verification_samples', 25)), 50)
        except ValueError:
            return 'Invalid input..'

        from functools import partial
        from multiprocessing import Process
        wrapped_single_model_creation = partial(single_model_creation, site_no, ls_resolution, dem_resolution, calibration_flag, validation_flag, sensitivity_flag, cal_pool_size, sen_pool_size, sen_total_evaluations, num_levels, max_cal_iterations, verification_samples)
        ## shoot off a process and forget about it
        process = Process(target=wrapped_single_model_creation) 
        process.start()
#        try:
#            if output_path := single_model_creation(site_no, ls_resolution, dem_resolution, calibration_flag, validation_flag, sensitivity_flag, cal_pool_size, sen_pool_size, sen_total_evaluations, num_levels, max_cal_iterations, verification_samples):
#               return send_file(output_path, as_attachment=True)
#            else:
#                output = f'Failed to create SWAT+ model for station {site_no}'
#        except Exception as e:
#            output = f'Error: {str(e)}'

    station_data = integrate_streamflow_data(r"/data/MyDataBase/SWATGenXAppData/USGS")
    station_list = station_data.SiteNumber.unique()
    return render_template('form_template.html', output=output, station_list=station_list)

@app.route('/get_station_characteristics', methods=['GET'])
def get_station_characteristics():
    station_no = request.args.get('station')
    station_data = integrate_streamflow_data(r"/data/MyDataBase/SWATGenXAppData/USGS")
    print("############ calling get_station_characteristics ###############")
    # Filter the data for the selected station
    station_row = station_data[station_data.SiteNumber == station_no]
    # Convert the filtered data to a dictionary with 'records' orientation, which makes a list of records
    characteristics_list = station_row.to_dict(orient='records')
    # Get the first record from the list (since there should only be one station with this number)
    characteristics = characteristics_list[0] if characteristics_list else {}

    if flat_characteristics := dict(characteristics.items()):
        print(f"Station {station_no} found")
        print(f"characteristics: {flat_characteristics}")
        return jsonify(flat_characteristics)
    else:
        return jsonify({"error": "Station not found"}), 404

if __name__ == '__main__':
    print("Starting server")
    serve(app, host='0.0.0.0', port=5500)
