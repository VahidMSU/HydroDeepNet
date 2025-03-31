from dataclasses import dataclass, field
import os
from typing import Optional
try:
    from ModelProcessing.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenXConfigPars import SWATGenXPaths
@dataclass
class ModelConfig:
    LEVEL: str
    VPUID: str
    NAME: str
    MODEL_NAME: str
    BASE_PATH: str = SWATGenXPaths.swatgenx_outlet_path
    START_YEAR: int = 1997
    END_YEAR: int = 2020
    nyskip: int = 3
    no_value: float = 1e6
    sensitivity_flag: bool = False
    calibration_flag: bool = False
    verification_flag: bool = False
    sen_total_evaluations: int = 1000
    sen_pool_size: int = 120
    num_levels: int = 10
    cal_pool_size: int = 50
    max_cal_iterations: int = 75
    termination_tolerance: float = 15
    epsilon: float = 0.001
    Ver_START_YEAR: int = 1997
    Ver_END_YEAR: int = 2020
    Ver_nyskip: int = 3
    range_reduction_flag: bool = False
    pet: str = 1
    cn: str = 1
    verification_samples: int = 5
    RESOLUTION: int = 250  # Default value
    yield_evaluation_flag: bool = False  # Default value
    et_evaluation_flag: bool = False  # Default value
    gw_elevation_evaluation_flag: bool = False  # Default value
    cms_to_cfs: float = 35.3147  # Default value
    bin_path: str = "/data/SWATGenXApp/codes/bin/"


    C1F: float = 0.5 
    C1I: float = 1 
    C2I: float = 0.5 
    C2F: float = 1
    Vmax: float  = 0.1 
    InertiaMin: float = 0.4 
    InertiaMax: float = 1

    # Paths to be initialized after other fields
    original_cal_file: Optional[str] = field(init=False)
    general_log_path: Optional[str] = field(init=False)
    model_base: Optional[str] = field(init=False)
    TxtInOut: Optional[str] = field(init=False)
    scenarios_path: Optional[str] = field(init=False)
    directory_path_si: Optional[str] = field(init=False)
    initial_points_path: Optional[str] = field(init=False)
    initial_values_path: Optional[str] = field(init=False)
    best_simulation_filename: Optional[str] = field(init=False)
    lake_path: Optional[str] = field(init=False)
    monthly_cal_figures_path: Optional[str] = field(init=False)
    daily_cal_figures_path: Optional[str] = field(init=False)
    calibration_figures_path: Optional[str] = field(init=False)
    monthly_sen_figures_path: Optional[str] = field(init=False)
    daily_sen_figures_path: Optional[str] = field(init=False)
    cal_file_path: Optional[str] = field(init=False)
    local_best_solutions_path: Optional[str] = field(init=False)
    gis_folder: Optional[str] = field(init=False)
    ver_perf_path: Optional[str] = field(init=False)

    def __post_init__(self):
        self.original_cal_file = os.path.join(SWATGenXPaths.bin_path, f'cal_parms_{self.MODEL_NAME}.cal')
        self.general_log_path = os.path.join(SWATGenXPaths.swatgenx_outlet_path, f'{self.VPUID}/{self.LEVEL}/log.txt')
        self.model_base = os.path.join(SWATGenXPaths.swatgenx_outlet_path, f'{self.VPUID}/{self.LEVEL}/{self.NAME}/')
        self.TxtInOut = os.path.join(self.model_base, f'{self.MODEL_NAME}/Scenarios/Default/TxtInOut/')
        self.scenarios_path = os.path.join(self.model_base, f'{self.MODEL_NAME}/Scenarios')
        self.directory_path_si = os.path.join(self.model_base, f'morris_Si_{self.MODEL_NAME}.csv')
        self.initial_points_path = os.path.join(self.model_base, f'initial_points_{self.MODEL_NAME}.csv')
        self.initial_values_path = os.path.join(self.model_base, f'initial_values_{self.MODEL_NAME}.csv')
        self.best_simulation_filename = os.path.join(self.model_base, f'best_solution_{self.MODEL_NAME}.txt')
        self.lake_path = os.path.join(self.model_base, f"{self.MODEL_NAME}/Watershed/Shapes/SWAT_plus_lakes.shp")
        self.monthly_cal_figures_path = os.path.join(self.model_base, f"figures_{self.MODEL_NAME}_calibration_monthly")
        self.daily_cal_figures_path = os.path.join(self.model_base, f"figures_{self.MODEL_NAME}_calibration_daily")
        self.calibration_figures_path = os.path.join(self.model_base, f"calibration_figures_{self.MODEL_NAME}")
        self.monthly_sen_figures_path = os.path.join(self.model_base, f"figures_{self.MODEL_NAME}_sensitivity_monthly")
        self.daily_sen_figures_path = os.path.join(self.model_base, f"figures_{self.MODEL_NAME}_sensitivity_daily")
        self.cal_file_path = os.path.join(self.model_base, f'cal_parms_{self.MODEL_NAME}.cal')
        self.local_best_solutions_path = os.path.join(self.model_base, f'local_best_solution_{self.MODEL_NAME}.txt')
        self.gis_folder = os.path.join(self.model_base, f'{self.MODEL_NAME}/gwflow_gis')
        self.ver_perf_path = os.path.join(self.model_base, f'recharg_output_{self.MODEL_NAME}/verification_performance_{self.MODEL_NAME}.txt')
        self.original_TxtInOut = f'{self.BASE_PATH}/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/Scenarios/Default/TxtInOut/'
        self.hru_new_target = f'{self.BASE_PATH}/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/Scenarios/Default/TxtInOut/hru.con'
        self.model_log_path = os.path.join(self.BASE_PATH, f"{self.VPUID}/{self.LEVEL}/{self.NAME}/")
        self.fig_files_paths = os.path.join(self.BASE_PATH, f'{self.VPUID}/{self.LEVEL}/{self.NAME}/figures_{self.MODEL_NAME}')
        self.streamflow_data_path = os.path.join(self.BASE_PATH, f"{self.VPUID}/{self.LEVEL}/{self.NAME}/streamflow_data/")
        