import pandas as pd

# Function to read the management.sch file
def read_management_file(file_path):
    # Read file
    with open(file_path, 'r') as file:
        content = file.readlines()
    
    # Initialize a list to hold all rows of the table
    management_data = []
    current_mgt = None

    # Iterate through the content and extract management details
    for line in content:
        if line.startswith('mgt_'):
            current_mgt = line.split()[0]
        elif current_mgt and line.strip():  # Capture operation details only if it's part of a management setting
            parts = line.split()
            if len(parts) >= 6:
                operation_type = parts[0]
                month = int(parts[1])  # Convert month to integer
                day = int(parts[2])    # Convert day to integer
                crop = parts[4] if len(parts) > 4 else None
                fertilizer_type = parts[5] if len(parts) > 5 else None
                amount = parts[6] if len(parts) > 6 else None
                
                # Append the row data to the management_data list
                management_data.append([current_mgt, operation_type, month, day, crop, fertilizer_type, amount])

    # Create a DataFrame to hold the results
    columns = ['Management Setting', 'Operation Type', 'Month', 'Day', 'Crop/Material', 'Fertilizer Type', 'Amount']
    return pd.DataFrame(management_data, columns=columns)

# Function to compute the number of years for each management setting
def calculate_years_of_operations(df):
    # Group by management setting and calculate the first and last operation by month/day
    df['Date'] = df['Month'] * 100 + df['Day']  # Create a combined "date" value for easy comparison
    year_span = df.groupby('Management Setting').agg(
        first_operation=('Date', 'min'),
        last_operation=('Date', 'max'),
        total_operations=('Date', 'count')
    ).reset_index()

    # Calculate the number of years based on how far apart the first and last operations are
    # Assuming operations start in one year and extend into the next, this is a simple approximation
    year_span['Years'] = ((year_span['last_operation'] - year_span['first_operation']) // 100).abs() + 1
    return year_span

# Function to summarize the management settings
def summarize_management_settings(df):
    # Group by management setting to understand the key differences between each one
    management_summary = df.groupby('Management Setting').agg({
        'Operation Type': lambda x: ', '.join(x.unique()),  # List of unique operation types
        'Month': 'count',  # Number of operations (to reflect intensity)
        'Crop/Material': lambda x: ', '.join(x.dropna().unique()),  # Unique crops/materials used
        'Fertilizer Type': lambda x: ', '.join(x.dropna().unique()),  # Unique fertilizer types used
        'Amount': lambda x: ', '.join(x.dropna().unique())  # Unique fertilizer amounts
    }).reset_index()

    # Renaming columns for better understanding
    management_summary.columns = ['Management Setting', 'Operations', 'Num Operations', 'Crops/Materials', 'Fertilizer Types', 'Amounts']
    return management_summary

# Function to analyze and find differences between management settings
def analyze_management_differences(summary_df, years_df):
    # Merge the management summary with the years of operations data
    combined_df = pd.merge(summary_df, years_df[['Management Setting', 'Years']], on='Management Setting')
    
    # Print the combined differences in management settings along with the number of years
    print("Management Setting Differences with Years of Operations:")
    print(combined_df)

if __name__ == "__main__":

    """
    This script reads the management.sch file and summarizes the management settings.
    It also calculates the number of years of operations for each management setting.
    
    """



    # File path to the management.sch file
    file_path = "/home/rafieiva/swatplus_installation/swatplus-61.0.2/data/Ames_sub1/management.sch"

    # Step 1: Read the management file
    management_df = read_management_file(file_path)

    # Step 2: Summarize the management settings
    management_summary = summarize_management_settings(management_df)

    # Step 3: Calculate the number of years of operations
    management_years = calculate_years_of_operations(management_df)

    # Step 4: Analyze and print the differences, including the number of years
    analyze_management_differences(management_summary, management_years)
