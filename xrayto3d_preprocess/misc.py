import pandas as pd
from typing import Dict,List,Optional
def range_inclusive(start, end):
     return range(start, end+1)

def write_csv(data:Dict[str,List],column_names:Optional[List[str]],file_path):
    """
    The dictionary should consist of key:List as key value pairs representing a single row.
    Optional name of each columns may also be provided
    """
    df = pd.DataFrame.from_dict(data, orient='index',
                       columns=column_names)
    df.to_csv(file_path)