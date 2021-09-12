import numpy as np
import pandas as pd
import pmdarima as pm


class ARMADL:
    def __init__(endog, exog=None):
        pass

    def model_selection():
        pass

    def process_distributed_lags():
        pass

    def estimate_exogenous_vars(exog, k=0, fill_val=0.0):
        """
        Generates distributed lags (DLs) for given exogenous variables.
        Args:
            exog: (pd.DataFrame) Dataframe containing the exogenous variables. 
                  If a list of lists or numpy array is provided, it will be converted to a pandas dataframe.
            k: (int or dict) Determines the number of lags in the individual exogenous variables.
               k can be integer, dict with integer or list of lags, for example:
               - k = 2 --> apply 0,1,2 lags to every variable
               - k = {'var1':2, 'var2':3, 'rest':1} --> apply 0,1,2 lags to var1, 0,1,2,3 lags to var2 and 0,1 lags to rest of the variables
               - k = {'var1':2, 'var2':[0,1,3], 'rest':1} --> apply 0,1,2 lags to var1, 0,1,3 lags to var2 and 0,1 lags to rest of the variables
            fill_val: (float or function) After applying lag, how to handle the missing values. Options are:
                      - A floating point number, eg. 0.0, np.nan, 1.5 etc.
                      - Any summary stat or aggregating function eg. np.mean, np.median (assumes numpy is imported as np).
        """
        if not isinstance(exog, pd.DataFrame):
            exog = pd.DataFrame(exog)

        final_exog = pd.DataFrame([])
        for var in exog.columns:
            if isinstance(k, int):
                tmp_k = k
            elif isinstance(k, dict):
                if var in k:
                    tmp_k = k[var]
                else:
                    tmp_k = k['rest']
            else:
                raise TypeError("k can only be of type integer or dict!")

            if isinstance(tmp_k, int):
                tmp_k = [*range(0, tmp_k+1)]

            for lag in tmp_k:
                final_exog[f'{var}_{lag}'] = exog[var].shift(lag,
                                                             fill_value=fill_val if isinstance(
                                                                 fill_val, float) else fill_val(exog[var])
                                                             )
        return final_exog
