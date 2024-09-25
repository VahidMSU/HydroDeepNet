
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    BASE_PATH = "/data/MyDataBase/CIWRE-BAE/SWAT_input/huc12"
    all_comparisons = []
    VPUIDS = os.listdir("/data/MyDataBase/SWATplus_by_VPUID")
    for vpuid in VPUIDS:
        if vpuid in ["0000", "0405", "0406"]:
            continue
        NAMES = os.listdir(f"/data/MyDataBase/SWATplus_by_VPUID/{vpuid}/huc12")
        NAMES.remove('log.txt')
        

        for name in NAMES:
            file = os.path.join("/data/MyDataBase/SWATplus_by_VPUID/", vpuid,"huc12" ,name, "climate_change_analysis/model_comparisions.csv")
            df = pd.read_csv(file)
            df["NAME"] = name
            all_comparisons.append(df)
    all_comparisons_df = pd.concat(all_comparisons)
    os.makedirs("climate_change", exist_ok=True)
    all_comparisons_df.to_csv(
        "climate_change/PRISM_LOCA2_model_comparisions.csv",
        index=False,
    )
    ## final dataframe columns: NAME,NSE,MSE,PBIAS,RMSE,ensemble,scenario,cc_model 
    # group by cc_model, scenario, ensemble and calculate mean of NSE, MSE, PBIAS, RMSE
    all_comparisons_df = all_comparisons_df.drop(columns='NAME').groupby(["cc_model", "scenario", "ensemble"]).median().reset_index()
    ## sort by RMSE from the lowest to the highest
    all_comparisons_df = all_comparisons_df.sort_values(by="RMSE")
    all_comparisons_df.to_csv(
        "climate_change/meadian_error_annual_PRISM_LOCA2_model_comparisions.csv",
        index=False,
    )
    ### only keep the best ensemble for each scenario
    all_comparisons_df = all_comparisons_df.groupby(["cc_model", "scenario"]).first().reset_index()
    all_comparisons_df.to_csv(
        "climate_change/meadian_error_annual_PRISM_LOCA2_model_comparisions_best_ensemble.csv",
        index=False,
    )


    ## sort by RMSE from the lowest to the highest
    all_comparisons_df = all_comparisons_df.sort_values(by="RMSE", ascending=False)




    ## plot the RMSE of each model in a h bar plot
    fig, ax = plt.subplots(figsize=(10, 15))
    all_comparisons_df.plot.barh(x='cc_model', y='RMSE', ax=ax)
    plt.xlabel("RMSE")
    plt.ylabel("Model")
    plt.title("RMSE of each model")
    plt.tight_layout()
    plt.grid(alpha=0.5, linestyle='--', linewidth=0.5)
    plt.savefig("climate_change/RMSE_of_each_model.png", dpi=300)
    plt.close()
    ## plot the RMSE of each model in a h bar plot
    