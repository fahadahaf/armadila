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
        self.fit_res = []

    def model_selection(self, strategy='split', train_size=0.9, accumulate_slide_win=False, **kwargs):
        if strategy == 'split':
            if isinstance(train_size, float):
                trn_size = int(train_size*len(self.endog))
            elif isinstance(train_size, int):
                trn_size = train_size
            else:
                raise TypeError(
                    "train_size can only be a float or an integer!")
            train_endog = self.endog.iloc[:trn_size]
            train_exog = self.exog.iloc[:trn_size]
            test_endog = self.endog.iloc[trn_size:]
            test_exog = self.exog.iloc[trn_size:]
            #print(len(train_endog), len(train_exog))
            fit_res = pm.auto_arima(train_endog, X=train_exog, **kwargs)
            train_preds = fit_res.predict(n_periods=trn_size, X=train_exog)
            test_preds = fit_res.predict(
                n_periods=test_endog.shape[0], X=test_exog)
            self.fit_res.append(fit_res)
            return train_endog, train_preds, test_endog, test_preds

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
        if exog is None:
            return exog

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
                var_name = f'{var}_{lag}' if lag > 0 else var
                final_exog[var_name] = exog[var].shift(lag,
                                                       fill_value=fill_val if isinstance(
                                                           fill_val, float) else fill_val(exog[var])
                                                       )
        return final_exog

    @staticmethod
    def pick_dl_granger_causality_test(endog, exog):
        pass

    def estimate_exogenous_vars(self):
        pass
