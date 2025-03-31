import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

def fetch_new_ranges(VPUID, NAME, LEVEL, BASE_PATH, MODEL_NAME):
    """ This function will estimate the new range of parameters
    and replace the default range in the calibration file"""
    
    local_best_solutions = os.path.join(BASE_PATH, fr"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/local_best_solution_{MODEL_NAME}.txt")
    if not os.path.exists(local_best_solutions):
        logging.error(f"{local_best_solutions} does not exist")
        return None
    df = pd.read_csv(local_best_solutions, delimiter=',')

    # Remove outliers based on the best score
    df = df[df['best_score'] < 100]

    # Exclude 'best_score' column for parameter distribution analysis
    parameters = df.columns.drop('best_score')
    def hypothesize_new_parameter_ranges(df, score_threshold=0, reduction_factor=0.9):
        new_ranges = {}

        #best_solutions = best_scores_df.sample(n=100)
        # instead of sampling, rank the best solutions and select the top 100
        best_solutions = df.sort_values(by='best_score').head(100) # this will assume that the best score is the lowest

        alpha = 0.5

        for parameter in df.columns.drop('best_score'):
            
            # Calculate the mean and standard deviation of the parameter values of the 100 best solutions

            mean = df[parameter].mean()
            std = df[parameter].std()

            original_range = (df[parameter].min(), df[parameter].max())

            # Set the new range based on the mean and standard deviation
            new_range = (mean - alpha * std, mean + alpha * std)

            if best_solutions is not None:
                # Check the new range with the 100 best solutions
                if best_solutions[parameter].min() < new_range[0]:
                    new_range = (best_solutions[parameter].min() - 0.1* abs(best_solutions[parameter].min()), new_range[1])
                if best_solutions[parameter].max() > new_range[1]:
                    new_range = (new_range[0], best_solutions[parameter].max() + 0.1* abs(best_solutions[parameter].max()))
            # if the new range is clode to the boundary
            new_ranges[parameter] = new_range

        return new_ranges

    # Hypothesize new parameter ranges and initial solutions
    new_ranges = hypothesize_new_parameter_ranges(df)
    # create a dataframe using new_ranges and save it. the names should be name, min, max
    new_ranges_df = pd.DataFrame(new_ranges.items(), columns=['name', 'range'])
    new_ranges_df[['min', 'max']] = pd.DataFrame(new_ranges_df['range'].tolist(), index=new_ranges_df.index)
    new_ranges_df.drop('range', axis=1, inplace=True)
    new_ranges_df.to_csv(os.path.join(BASE_PATH,fr"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/new_ranges.csv"), index=False)
    cal_parms = pd.read_csv(os.path.join(BASE_PATH,fr"bin/cal_parms_{MODEL_NAME}.cal"),skiprows=1, delimiter='\s+')  
    new_cal_parms = pd.merge(cal_parms.drop(columns=['min', 'max']), new_ranges_df, on='name', how='left')

    with open (os.path.join(BASE_PATH,fr"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/new_cal_parms_{MODEL_NAME}.cal"), 'w') as file:
        file.write("calibration parameters with new range\n")
    new_cal_parms =  new_cal_parms[['name', 'file_name' , 'min','max' ,'operation']]
    new_cal_parms.to_csv(os.path.join(BASE_PATH,fr"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/new_cal_parms_{MODEL_NAME}.cal"), index=False, mode ='a', sep='\t')
# Plot best score vs. each parameter with new hypothesized ranges
    for parameter in parameters:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[parameter], df['best_score'], alpha=0.5)
        
        # Draw vertical lines to show the new hypothesized ranges of the parameter
        new_range = new_ranges[parameter]
        plt.axvline(new_range[0], color='g', linestyle='dashed', linewidth=2, label='New Range Low')
        plt.axvline(new_range[1], color='g', linestyle='dashed', linewidth=2, label='New Range High')
        
        plt.title(f"Best Score vs. {parameter} {MODEL_NAME}")
        plt.xlabel(parameter)
        plt.ylabel('Best Score')
        
        plt.legend()
        plt.grid(True)
        os.makedirs(f"/data/SWATGenX/Users/{username}/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/parameter_vs_best_score/", exist_ok=True)
        plt.savefig(f"/data/SWATGenX/Users/{username}/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/parameter_vs_best_score/{parameter}_vs_best_score_{MODEL_NAME}.jpeg", dpi= 300)
        plt.close()

    return new_ranges


