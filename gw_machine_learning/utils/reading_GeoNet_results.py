import os 
import pandas as pd
results = []
if os.path.exists("results.csv"):
    os.remove("results.csv")

files = os.listdir("report")
for file in files:
    if file.endswith(".txt") and "feature" not in file and "FFR" in file :
        with open(os.path.join("report", file), "r") as f:
            # line format: Test_obs_V_COND_2_50m, MSE: 2197.19, RMSE: 46.87, NSE: 0.37, R2: 0.42, MPE: -6182.85
            # read the line and split it by comma and then split by :
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split(",")
                stage = line[0].split("_")[0]
                resolution = line[0].split("_")[-1]
                name = line[0].split("_")[2:-1]
                name = "_".join(name)

                mse = float(line[1].split(":")[1].strip())
                rmse = float(line[2].split(":")[1].strip())
                nse = float(line[3].split(":")[1].strip())
                r2 = float(line[4].split(":")[1].strip())
                mpe = float(line[5].split(":")[1].strip())
                results.append([stage,resolution,name, mse, rmse, nse, r2, mpe])

print(results)
# create a pandas dataframe from the results

df = pd.DataFrame(results, columns=["stage","resolution","name", "MSE", "RMSE", "NSE", "R2", "MPE"])
# drop 50 and 100 resolution
df = df[df["resolution"] != "50m"]
df = df[df["resolution"] != "100m"]
df.sort_values(by=["stage","name"]).reset_index(drop=True).to_csv("report/FFR_results.csv", index=False)
