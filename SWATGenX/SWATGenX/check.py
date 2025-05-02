import os
from collections import Counter
import numpy as np
cli_path = "/data/SWATGenXApp/Users/admin/SWATplus_by_VPUID/0408/huc12/04155500/PRISM/slr.cli"

with open(cli_path, 'r') as file:
    lines = file.readlines()[2:]
    ## find duplicate lines
    lines = [line.strip() for line in lines]
    print(f"number of lines: {len(lines)}")
    ## find duplicate lines
    duplicates = [item for item, count in Counter(lines).items() if count > 1]
    print(f"number of duplicates: {len(duplicates)}")