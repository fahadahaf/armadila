import numpy as np
import pandas as pd
import pickle
import pmdarima as pm

from statsmodels.tsa.stattools import grangercausalitytests


class ARMADL:
    def __init__(self, endog, exog=None, dl_param=None, fill_val=0.0):
        """
        ARMADL class, ARMA/ARIMA with distributed lags.
        Args:
            endog: (pd.Series) the target time series.
            exog: (pd.DataFrame) Exogenous variable(s).
            dl_param: (int or dict) Distributed lags parameter, see generate_distributed_lags() for details.
            fill_val: (float or function) fill values for shifted series, see generate_distributed_lags() for details.
        """
        self.endog = endog
        self.exog = exog
        self.dl_param = dl_param if not isinstance(
            dl_param, str) else self.pick_dl_granger_causality_test(self.endog, self.exog)
        self.fill_val = fill_val
        if self.dl_param is not None:
            self.exog = self.generate_distributed_lags(
                self.exog, k=self.dl_param, fill_val=self.fill_val)
        self.fit_res = None
        self.best_params = None

    def model_selection(self, strategy='split', train_size=0.9, accumulate_slide_win=False, get_results=True, use_pretrained=False, store_model=True, model_path=None, **auto_arima_args):
        """
        Model selection with distributed lags, a wrapper for the auto_arima function.
        Args:
            strategy: (str) To use regular train/test split for model selection or the sliding window approach (TO-DO). Default: `split`
            train_size: (int or float) For regular train/test split, size of the train set or fraction of the dataset to use for training.
                        In the later case, the remaining fraction of data will be used for testing. Default: 0.9
            accumulate_slide_win: (bool) Whether to use the accumulated sliding window instead of a fixed size (TO-DO). Default: False
            get_results: (bool) Whether to get the final model selection results. Default: True
            use_pretrained: (bool) Whether to use pretrained model (post model selection). Default: True
            store_model: (bool) Whether to store the best model post model selection. Default: True
            model_path: (str) Path (with filename) for the model to store. Default: None
            **auto_arima_args: (dict) arguments for the auto_arima function. 
                               This should include everything except the endogenous and exogenous variable(s).
        Returns:
            (tuple): the train and test ground truth and predictions.
        """
        self.fit_res = []
        self.best_params = []
        if use_pretrained or store_model:
            if model_path is None:
                raise TypeError(
                    "model_path cannot be None if either of use_pretrained or store_model are set to True!")

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
            if use_pretrained:
                try:
                    with open(model_path, 'rb') as f:
                        fit_res = pickle.load(f)[0]
                except:
                    print(
                        "Cannot load the pretrained model. Set use_pretrained=False to redo the model selection.")
            else:
                fit_res = pm.auto_arima(
                    train_endog, X=train_exog, **auto_arima_args)
            try:
                train_preds = fit_res.predict(n_periods=trn_size, X=train_exog)
                test_preds = fit_res.predict(
                    n_periods=test_endog.shape[0],
                    X=test_exog)
            except:
                print("Something went wrong! if you're using a pretrained model, make sure the same set of auto_arima parameters are used.")
            if len(self.fit_res) == 0:
                self.fit_res.append(fit_res)
                self.best_params.append(
                    list(fit_res.order) + [fit_res.seasonal_order[-1]])

            if store_model and use_pretrained == False:
                with open(model_path, 'wb') as f:
                    pickle.dump(self.fit_res, f)

            if get_results:
                return train_endog, train_preds, test_endog, test_preds
            else:
                return None

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
        Returns:
            final_exog: (pd.DataFrame) a dataframe of exogenous variables with distributed lags.
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
    def pick_dl_granger_causality_test(endog, exog, maxlag=8, htest='ssr_ftest', verbose=False, lagthresh=8, alpha=0.05):
        """
        Picks distributed lags for exogenous variables based on granger causality test.
        Args:
            endog: (pd.Series) The endogenous data, or target series which is to be forecasted.
            exog: (pd.DataFrame) The exogenous data.
            maxlag: (int) Number of lags to consider in the exogenous variable; this is one of the arguments for the Granger Causality test. Default: 8
            htest: (str) Hypothesis test to consider; this is one of the arguments for the Granger Causality test. Default: 'ssr_ftest'
            verbose: (boolean) Verbose argument of the Granger Causality test function. Default: False
            lagthresh: (int) Number of lags threshold; the best lag suggested by the test should be less than or equal to this value. Default: 8
            alpha: (float) The p-value alpha cutoff to use in the Granger Causality test. Default: 0.05
        Returns:
            dict: A dictionary with keys representing the exogenous variable(s) and values as lists, in the following format: [lag_pvalues, all_accepted_lags, best_lag]. 
                  The best_lag is the suggested lag to use for the corresponding exogenous variable.

        """
        exog_res = {}
        for col in exog.columns:
            data = pd.DataFrame((endog, exog[col])).T
            res_dict = grangercausalitytests(data.dropna(), maxlag=maxlag, verbose=verbose)
            lag_pvals = [res_dict[i][0][htest][1] for i in range(1, maxlag+1)]
            lag_pvals = np.asarray(lag_pvals)
            all_lags = np.argwhere(lag_pvals < alpha).flatten() + 1 #lags start at 1 not 0
            best_lag = None if len(all_lags)==0 else all_lags[0]
            if best_lag is not None:
                best_lag = None if best_lag > lagthresh else best_lag
            exog_res[col] = [lag_pvals, all_lags, best_lag]
        return exog_res

    def estimate_exogenous_vars(self):
        """
        Estimates exogenous variables when not available at forecast/prediction time.
        **TO-DO**
        """
        pass
