import numpy as np
import pandas as pd
import pmdarima as pm


class ARMADL:
    def __init__(self, endog, exog=None, dl_param=None, fill_val=0.0):
        self.endog = endog
        self.exog = exog
        self.dl_param = dl_param if not isinstance(
            dl_param, str) else self.pick_dl_granger_causality_test(self.endog, self.exog)
        self.fill_val = fill_val
        if self.dl_param is not None:
            self.exog = self.generate_distributed_lags(
                self.exog, k=self.dl_param, fill_val=self.fill_val)

    def model_selection(self, strategy='split', train_size=0.8, accumulate_slide_win=False, **kwargs):
        fit_res = pm.auto_arima(self.endog, X=self.exog, **kwargs)

    @staticmethod
    def generate_distributed_lags(exog, k=0, fill_val=0.0):
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

    @staticmethod
    def pick_dl_granger_causality_test(endog, exog):
        pass

    def estimate_exogenous_vars(self):
        pass
