import pandas as pd
import numpy as np

# --- Data from a formula ---
"""
Used for more advanced models, such as polynomial fits, in which the data may 
contain the variables x1, x2, ..., but the formula is: y ~ x1 + x2^2 + x2.
"""

# Using numpy: pandas is only used in case a pandas data frame is used as input.
# Personally, I don't like this function's approach for two reasons:
# - Every variable must be of the same type (numpy's trait), so it won't be
#   useful for more advanced models.
# - Pandas trivializes working with variable names.
# 
# Consequently, I made a pandas version below, and is the one I'll be using — most likely.

def numpy_array_from_formula(formula: str, data = None, var_names = None):
    # Initial check
    if type(data) is pd.DataFrame:
        var_names = data.columns
        data = data.values

    var_names = np.array(var_names, dtype=str)
    data = np.array(data)

    # Format formula
    f = formula.replace(' ', '')

    # Get "~" position -> get target and independent variables
    idx = f.find('~')
    target = f[:idx]
    ind_vars = f[idx+1:].split(sep='+') # Split independent variables into a list
    ind_vars.sort()

    # Prepare training data with variable name list
    df = np.zeros(shape=(data.shape[0], len(ind_vars)+1)) # initialize output
    df[:, 0] = data[:, target==var_names][0] # first column (output) = target variable
    vars_output = [target]

    for idx, var in enumerate(ind_vars, start=1):
        if '^' in var:
            # If poly: find varname and power
            original_name, power = var.split('^')
            df[:, idx] = np.power(data[:, original_name==var_names][0], int(power))
        else:
            df[:, idx] = data[:, var==var_names][0]
        
        # Store independent variable name
        vars_output.append(var)

    # Return data and variable names
    return (df, vars_output)

# Formulas with pandas: makes more sense when working with different type of data, as numpy array
# require that all values are the same type. Also, easier to work with variable names.
def pandas_df_from_formula(formula: str, data = pd.DataFrame):
    # --- Instructions from the formula ---
    # Format formula
    f = formula.replace(' ', '')

    # Get "~" position 
    idx = f.find('~')
    target = f[:idx]
    ind_vars = f[idx+1:].split(sep='+') # Split independent variables into a list
    ind_vars.sort()

    # Create a list of all variables (new columns of output)
    all_vars = [target]
    all_vars.extend(ind_vars)

    # --- Prepare training data ---
    # Initialize output
    df = pd.DataFrame(
        np.zeros(shape=(data.shape[0], len(ind_vars)+1)),
        columns=all_vars
    )

    # Store data in new data frame
    for var in all_vars:
        if '^' in var:
            # If poly: find varname and power
            original_name, power = var.split('^')
            df.loc[:, var] = np.power(data.loc[:, original_name].values, int(power))
        else:
            df.loc[:, var] = data.loc[:, var].values

    # Return data and variable names
    return df



# Example data
N = 1000
y = np.random.random(size=N)
data = pd.DataFrame.from_dict(
    {
        'y': y,
        'x1': np.random.randint(low=0, high=5, size=N),
        'x2': np.sqrt(y)
    }
)

formula = 'y~ x1+ x2^2 +  x2'

a = pandas_df_from_formula(formula=formula, data=data)
print(a.head())