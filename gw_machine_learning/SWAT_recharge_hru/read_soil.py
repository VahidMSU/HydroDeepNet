import pandas as pd
import re
def get_soil_data(NAME= "04111379", VPUID="0000"):
    file_path = f'/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut/soils.sol'

    # Read the file content
    with open(file_path, 'r') as file:
        file_content = file.readlines()
        df = pd.DataFrame(columns=['soil_id', 'lyr', 'dp', 'bd', 'awc', 'soil_k', 'carbon', 'clay', 'silt'])
        row = 0
        for i, line in enumerate(file_content):
            if line.startswith('name'):
                headers = line.split()
            if re.match(r'\d', line):
                parts = line.split()
                soil_id = parts[0]
                nly = int(parts[1])
                for j in range(nly):
                    next_line = file_content[i+j+1].split()  # Corrected to access the next lines correctly
                    df.loc[row, 'soil_id'] = soil_id
                    df.loc[row, 'lyr'] = j+1
                    df.loc[row, 'dp'] = next_line[1]
                    df.loc[row, 'bd'] = next_line[2]
                    df.loc[row, 'awc'] = next_line[3]
                    df.loc[row, 'soil_k'] = next_line[4]
                    df.loc[row, 'carbon'] = next_line[5]
                    df.loc[row, 'clay'] = next_line[6]
                    df.loc[row, 'silt'] = next_line[7]
                    row += 1

    ## group by soil id and get the average of each column
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.groupby('soil_id').mean().reset_index()
    print(df)
    return df

if __name__ == '__main__':
    df = get_soil_data()