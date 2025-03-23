from MODGenX.MODGenXCore import MODGenXCore
from MODGenX.config import MODFLOWGenXPaths
from MODGenX.path_handler import PathHandler
import sys
import argparse
import os
sys.path.append('/data/SWATGenXApp/codes/SWATGenX/')
from SWATGenX.utils import find_VPUID
"""
/***************************************************************************
    SWATGenX
                            -------------------
        begin                : 2023-05-15
        copyright            : (C) 2024 by Vahid Rafiei
        email                : rafieiva@msu.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

def create_modflow_model(username, NAME, VPUID=None, BASE_PATH=None, LEVEL=None, 
                         RESOLUTION=None, MODEL_NAME=None, ML=False, SWAT_MODEL_NAME=None,
                         n_sublay_1=None, n_sublay_2=None, k_bedrock=None, 
                         bedrock_thickness=None, fit_to_meter=None, headtol=None,
                         fluxtol=None, maxiterout=None):
    """
    Create a MODFLOW model with the given parameters.
    
    Parameters:
    -----------
    username : str
        Username for file paths
    NAME : str
        Name of the model (usually HUC ID)
    VPUID : str, optional
        VPUID identifier, will be derived from NAME if not provided
    BASE_PATH : str, optional
        Base path for data
    LEVEL : str, optional
        Level of analysis (huc12, huc8, etc.)
    RESOLUTION : int, optional
        Resolution in meters
    MODEL_NAME : str, optional
        Name of the MODFLOW model
    ML : bool, optional
        Whether to use machine learning predictions
    SWAT_MODEL_NAME : str, optional
        Name of the SWAT model
    n_sublay_1 : int, optional
        Number of sublayers in layer 1
    n_sublay_2 : int, optional
        Number of sublayers in layer 2
    k_bedrock : float, optional
        Hydraulic conductivity of bedrock
    bedrock_thickness : float, optional
        Thickness of bedrock layer
    fit_to_meter : float, optional
        Conversion factor from feet to meters
    headtol : float, optional
        Head change tolerance for convergence
    fluxtol : float, optional
        Flux tolerance for convergence
    maxiterout : int, optional
        Maximum number of outer iterations
        
    Returns:
    --------
    MODGenXCore
        The MODFLOW model object
    """
    # Set defaults for any missing parameters
    if VPUID is None:
        VPUID = find_VPUID(NAME)
    
    if BASE_PATH is None:
        BASE_PATH = "/data/SWATGenXApp/GenXAppData/"
    
    if LEVEL is None:
        LEVEL = 'huc12'
    
    if RESOLUTION is None:
        RESOLUTION = 250
    
    if MODEL_NAME is None:
        MODEL_NAME = f'MODFLOW_{RESOLUTION}m'
    
    if SWAT_MODEL_NAME is None:
        SWAT_MODEL_NAME = 'SWAT_MODEL_Web_Application'
    
    # Create configuration object with all parameters
    config = MODFLOWGenXPaths(
        username=username,
        BASE_PATH=BASE_PATH,
        MODFLOW_MODEL_NAME=MODEL_NAME,
        SWAT_MODEL_NAME=SWAT_MODEL_NAME,
        LEVEL=LEVEL,
        VPUID=VPUID,
        NAME=NAME,
        RESOLUTION=RESOLUTION
    )
    
    # Update optional parameters if provided
    if n_sublay_1 is not None:
        config.n_sublay_1 = n_sublay_1
    if n_sublay_2 is not None:
        config.n_sublay_2 = n_sublay_2
    if k_bedrock is not None:
        config.k_bedrock = k_bedrock
    if bedrock_thickness is not None:
        config.bedrock_thickness = bedrock_thickness
    if fit_to_meter is not None:
        config.fit_to_meter = fit_to_meter
    if headtol is not None:
        config.headtol = headtol
    if fluxtol is not None:
        config.fluxtol = fluxtol
    if maxiterout is not None:
        config.maxiterout = maxiterout
    
    # Create path handler
    path_handler = PathHandler(config)
    
    # Ensure critical directories exist
    os.makedirs(path_handler.get_model_path(), exist_ok=True)
    os.makedirs(path_handler.get_raster_input_dir(), exist_ok=True)
    
    # Create and run the model
    modflow_model = MODGenXCore(
        username, NAME, VPUID, BASE_PATH, LEVEL, RESOLUTION, 
        MODEL_NAME, ML, SWAT_MODEL_NAME, config
    )
    
    modflow_model.create_modflow_model()
    return modflow_model

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create MODFLOW model with SWATGenX')
    parser.add_argument('--username', required=False, help='Username for file paths', default='vahidr32')
    parser.add_argument('--name', required=False, help='Name of the model (usually HUC ID)', default="04112500")
    parser.add_argument('--vpuid', help='VPUID identifier, will be derived from NAME if not provided', default="0405")
    parser.add_argument('--base_path', default="/data/SWATGenXApp/GenXAppData/", help='Base path for data')
    parser.add_argument('--level', default='huc12', help='Level of analysis (huc12, huc8, etc.)')
    parser.add_argument('--resolution', type=int, default=250, help='Resolution in meters')
    parser.add_argument('--model_name', help='Name of the MODFLOW model', default='MODFLOW_250m')
    parser.add_argument('--ml', action='store_true', help='Use machine learning predictions')
    parser.add_argument('--swat_model_name', default='SWAT_MODEL_Web_Application', help='Name of the SWAT model')
    parser.add_argument('--n_sublay_1', type=int, help='Number of sublayers in layer 1')
    parser.add_argument('--n_sublay_2', type=int, help='Number of sublayers in layer 2')
    parser.add_argument('--k_bedrock', type=float, help='Hydraulic conductivity of bedrock')
    parser.add_argument('--bedrock_thickness', type=float, help='Thickness of bedrock layer')
    parser.add_argument('--fit_to_meter', type=float, help='Conversion factor from feet to meters')
    parser.add_argument('--headtol', type=float, help='Head change tolerance for convergence')
    parser.add_argument('--fluxtol', type=float, help='Flux tolerance for convergence')
    parser.add_argument('--maxiterout', type=int, help='Maximum number of outer iterations')
    
    args = parser.parse_args()
    
    # If model_name is not provided, derive it from resolution
    if args.model_name is None:
        args.model_name = f'MODFLOW_{args.resolution}m'
    
    # Create and run the model with all the provided arguments
    modflow_model = create_modflow_model(
        args.username, args.name, args.vpuid, args.base_path, args.level,
        args.resolution, args.model_name, args.ml, args.swat_model_name,
        args.n_sublay_1, args.n_sublay_2, args.k_bedrock, args.bedrock_thickness,
        args.fit_to_meter, args.headtol, args.fluxtol, args.maxiterout
    )
