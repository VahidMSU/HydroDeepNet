# sourcery skip: swap-if-else-branches, use-named-expression
from ModelProcessing.cc_evaluation import SWAT_cc_controller

if __name__ == "__main__":
		parallel = True

		config = {
				"CC_SCENARIO": "MPI-ESM1-2-HR_ssp",
				"BASE_PATH": "/data/MyDataBase/SWATplus_by_VPUID",
				"start_year": 2015,
				"end_year": 2100,
				"model_name": "SWAT_gwflow_MODEL",
				"num_processes": 80,
				"stage": "projection",
				"convert2h5": True,
				'clean_up_cc_model': True,
				"filter_vpuid":['0406'],  ### note: the analysis will be performed only for the VPUID in this list
				"rewrite": False,
				"hru_wb": "hru_wb    n    y    y    y\n",
				"change_sd": "change_sd    n    y    y    y\n",
		}

		SWAT_cc_controller(config=config).process()

