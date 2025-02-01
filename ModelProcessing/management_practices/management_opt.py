def generate_optimized_management_schedule(num_cycles, output_file):
    """
    Generates an optimized management schedule for Michigan with a double-cropping system (corn and soybeans in the same year),
    optimized for fertilizer, irrigation, and tillage timing, and repeats the one-year double-cropping system for the specified number of cycles.

    Args:
    - num_cycles: Number of one-year cycles to generate (corn followed by soybean).
    - output_file: Path to the output file where the schedule will be written.
    """
    # Define the optimized management operations for Corn (Spring-Summer)
    management_operations_corn = [
        # Tillage for corn in early April
        "                                                    till          4          1    0.0     fldcult        null     0.000\n",
        # Planting corn in late April
        "                                                    plnt          4         25    0.0        corn        null         0\n",
        # Pre-plant fertilizer: Phosphorus and Potassium
        "                                                    fert          4         20    0.0       elem_p       null      0.000\n",
        "                                                    fert          4         20    0.0        11_52_00    null     120.000\n",
        # Nitrogen (30%) at planting
        "                                                    fert          5          1    0.0       elem_n       null   30.000\n",
        # Irrigation starting in early June
        "                                                    irrm          6          1    0.0    sprinkler_med   null       0.0\n",
        # Side-dress Nitrogen (60%) in June
        "                                                    fert          6         10    0.0       elem_n       null   60.000\n",
        # Additional irrigation in July during grain filling stage
        "                                                    irrm          7          5    0.0    sprinkler_med   null       0.0\n",
        "                                                    irrm          7         20    0.0    sprinkler_med   null       0.0\n",
        # Harvest corn in late August
        "                                                    hvkl          8         25    0.0        corn       grain     0.000\n",
        # Post-harvest fertilizer for soybean planting
        "                                                    fert          9          1    0.0       05_10_15    null     100.000\n"
    ]

    # Define the optimized management operations for Soybean (Late Summer-Winter)
    management_operations_soybean = [
        # Tillage for soybean immediately after corn harvest
        "                                                    till          9          5    0.0     fldcult        null     0.000\n",
        # Planting soybean in early September
        "                                                    plnt          9          10    0.0       soybn       null         0\n",
        # Fertilizer before planting: Potassium and low Nitrogen for soybean
        "                                                    fert          9          8    0.0       elem_p       null      0.000\n",
        "                                                    fert          9          8    0.0        null    11_52_00   100.000\n",
        # Nitrogen at planting (reduced rate for soybean)
        "                                                    fert          9         15    0.0       elem_n       null   20.000\n",
        # Irrigation in early October during pod filling stage
        "                                                    irrm         10          1    0.0    sprinkler_med   null       0.0\n",
        "                                                    irrm         10         20    0.0    sprinkler_med   null       0.0\n",
        # Harvest soybean in late November
        "                                                    hvkl         11         30    0.0       soybn      grain     0.000\n",
        # Post-harvest fertilizer for next year's corn planting
        "                                                    fert         12          1    0.0       05_10_15    null     100.000\n",
        # Skip operation (end of year)
        "                                                    skip          0          0    0.0        null        null         0\n"
    ]

    # Calculate the total number of operations per cycle (corn + soybean)
    total_operations_per_cycle = len(management_operations_corn) + len(management_operations_soybean)
    total_operations = total_operations_per_cycle * num_cycles

    # Open the output file for writing
    with open(output_file, 'w') as f:
        # Write the header for the management file, including the correct NUMB_OPS
        f.write("management.sch file AMES\n")
        f.write(f"                          NAME NUMB_OPS NUMB_AUTO OP_TYP        MON        DAY HU_SCH   OP_DATA1    OP_DATA2  OP_DATA3\n")
        f.write(f"mgt_01                              {total_operations}         0\n")

        # Write the corn and soybean management schedule for the given number of cycles
        for cycle in range(num_cycles):
            # Year 1: Corn management
            for operation in management_operations_corn:
                f.write(operation)
            
            # Year 1: Soybean management
            for operation in management_operations_soybean:
                f.write(operation)

# Call the function to generate the optimized schedule for 2 cycles (4 years in total)
output_file_path = "/home/rafieiva/MyDataBase/codebase/ModelProcessing/example_inputs/management.sch"
generate_optimized_management_schedule(20, output_file_path)
