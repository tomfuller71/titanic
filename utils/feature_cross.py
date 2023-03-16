import pandas as pd
import numpy as np
from itertools import product

def cross_category_features(
    data: pd.DataFrame,
    cross: list[str],
    remove_originals: bool = True
) -> pd.DataFrame:
    """
    Add feature crosses to the  based on the columns in cross_cols.  The columns must have already been factorized / ordinal encoded.

    :param data: The data to add feature crosses to
    :param cross_cols: The columns to cross. Columns must be int categorical 0 to n-1
    :param remove_originals: If True, remove the original columns from the data

    :return: The data with the feature crosses added
    """
    
    def set_hot_index(row):
        hot_index = (row[cross] * offsets).sum()
        row[hot_index + org_col_len] = 1
        return row
    
    org_col_len = data.shape[1]

    str_values = [[col + str(val) for val in sorted(data[col].unique())]
                  for col in cross]
    
    cross_names = ["_".join(x) for x in product(*str_values)]
    cross_features = pd.DataFrame(
        data=np.zeros((data.shape[0], len(cross_names))),
        columns=cross_names,
        dtype="int64")
    
    data = pd.concat([data, cross_features], axis=1)
    
    max_vals = data[cross].max(axis=0) + 1
    offsets = [np.prod(max_vals[i+1:]) for i in range(len(max_vals))]
    data.apply(set_hot_index, axis=1)

    if remove_originals:
        data = data.drop(columns=cross)

    return data