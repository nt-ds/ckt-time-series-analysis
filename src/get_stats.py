#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as smapi
import statsmodels.graphics.tsaplots as tsaplots
import statsmodels.tsa as tsa
import warnings

from pylab import rcParams
from sklearn.metrics import mean_squared_error

sns.set_style("whitegrid")
warnings.filterwarnings("ignore")


# In[2]:


class Stats():
    
    @staticmethod
    def plot_time_series(data, title):
        plt.figure(figsize=(20, 10))
        plt.plot(data)
        plt.title(title)
        plt.show()
    
    
    @staticmethod
    def demean(data, *args, **kwargs):
        dem_data = data.subtract(data.mean())
        return dem_data
    
    
    @staticmethod
    def adfuller(data, *args, **kwargs):
        # method to run augmented dickey-fuller test for unit-root
        # if we can reject the null hypothesis, our series is probably (weakly) stationary
        # the default is that the series is to run with a constant (i.e. non-zero mean)
        # but we can demean the series first
        # we can put a max-lag after the fact
        # note that lag convention is pythonic i.e. lag=0 means an AR(1)
        
        def_args = {}
        def_args['maxlag'] = kwargs.get('maxlag', None)
        def_args['regression'] = kwargs.get('regression', 'c')
        if def_args['maxlag'] is None:
            def_args['autolag'] = kwargs.get('autolag', 't-stat')
        else:
            def_args['autolag'] = None
        def_args['store'] = kwargs.get('store', False)
        def_args['regresults'] = kwargs.get('regresults', True)
        
        result = tsa.stattools.adfuller(data.values.reshape(-1,), **def_args)
        
#         print('ADF Statistic: %f' % result[0])
#         print('p-value:       %f' % result[1])
#         print('Critical Values:')
#         for key, value in result[4].items():
#             print('\t%s: %.3f' % (key, value))
#         print('Used Lag:      %d' % result[2])
        
        ad_results = {'ADF Statistic': result[0], 'p-value': result[1], 'Used Lag': result[-1].usedlag+1}
        ad_results.update(result[2])
        return pd.Series(ad_results)
    
    
    @staticmethod
    def seasonal_decompose(data, *args, **kwargs):
        # return a naive seasonal decomposition using moving averages
        
        def_args = {}
        def_args['model'] = kwargs.get('model', 'additive')
        def_args['filt'] = kwargs.get('filt', None)
        def_args['freq'] = kwargs.get('freq', None)
        def_args['two_sided'] = kwargs.get('two_sided', True)
        def_args['extrapolate_trend'] = kwargs.get('extrapolate_trend', 0)
        
        result = tsa.seasonal.seasonal_decompose(data, **def_args)
        rcParams['figure.figsize'] = 20, 10
        result.plot()
        plt.show()
        
        
    @staticmethod
    def plot_acf(data, *args, **kwargs):
        # plot the autocorrelation function
        # lags on the horizontal axis
        # the correlations on the vertical axis
        
        def_args = {}
        def_args['ax'] = kwargs.get('ax', None)
        def_args['lags'] = kwargs.get('lags', None)
        def_args['alpha'] = kwargs.get('alpha', 0.05)
        def_args['use_vlines'] = kwargs.get('use_vlines', True)
        def_args['unbiased'] = kwargs.get('unbiased', False)
        def_args['fft'] = kwargs.get('fft', False)
        def_args['title'] = kwargs.get('title', 'Autocorrelation')
        def_args['zero'] = kwargs.get('zero', True)
        def_args['vlines_kwargs'] = kwargs.get('vlines_kwargs', None)
        
        tsaplots.plot_acf(data, **def_args)
        plt.show()
        
        
    @staticmethod
    def plot_pacf(data, *args, **kwargs):
        # plot the partial autocorrelation function
        # lags on the horizontal axis
        # the correlations on the vertical axis
        
        def_args = {}
        def_args['ax'] = kwargs.get('ax', None)
        def_args['lags'] = kwargs.get('lags', None)
        def_args['alpha'] = kwargs.get('alpha', 0.05)
        def_args['method'] = kwargs.get('method', 'ywunbiased')
        def_args['use_vlines'] = kwargs.get('use_vlines', True)
        def_args['title'] = kwargs.get('title', 'Partial Autocorrelation')
        def_args['zero'] = kwargs.get('zero', True)
        def_args['vlines_kwargs'] = kwargs.get('vlines_kwargs', None)
        
        tsaplots.plot_pacf(data, **def_args)
        plt.show()
    
    
    @staticmethod
    def train_test_split(data):
        train, test = data[:len(data)-33], data[len(data)-33:]
        return train, test
    
    
    @staticmethod
    def AR_model(data, *args, **kwargs):
        train, test = Stats.train_test_split(data)
        
        def_args_model = {}
        def_args_model['dates'] = kwargs.get('dates', None)
        def_args_model['freq'] = kwargs.get('freq', None)
        def_args_model['missing'] = kwargs.get('missing', None)
        model = tsa.ar_model.AR(train, **def_args_model)
        
        def_args_fit = {}
        def_args_fit['maxlag'] = kwargs.get('maxlag', None)
        def_args_fit['ic'] = kwargs.get('ic', 'aic')
        def_args_fit['trend'] = kwargs.get('trend', 'nc')
        model_fit = model.fit(**def_args_fit)
        
        predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
        error = mean_squared_error(test, predictions)
        
        plt.figure(figsize=(20, 10))
        plt.plot(test, label='Expected')
        plt.plot(predictions, color='red', label='Predicted')
        plt.title(f'AR({model_fit.k_ar}); 33-Month Forecast; RMSE = %.3f; AIC = %.3f' % (np.sqrt(error), model_fit.aic))
        plt.legend(loc='best')
        plt.show()
    
    
    @staticmethod
    def ARMA_model(data, maxp=3, maxq=3, *args, **kwargs):
        train, test = Stats.train_test_split(data)
        
        def_args_fit = {}
        def_args_fit['trend'] = kwargs.get('trend', 'nc')
        
        plags = range(maxp+1)
        qlags = range(maxq+1)
        p_q = [(x, y) for x in plags for y in qlags]
        all_models = {'ARMA({0},{1})'.format(x[0], x[1]): tsa.arima_model.ARMA(train, x) for x in p_q}
        
        models = []
        preds = []
        errors = []
        aics = []
        
        for model in all_models:
            try:
                model_fit = all_models[model].fit(**def_args_fit)
                aic = model_fit.aic
                predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
                aics.append(aic)
                models.append(model)
                preds.append(predictions)
                error = mean_squared_error(test, predictions)
                errors.append(np.sqrt(error))
            except:
                pass
            
        df = pd.DataFrame(zip(models, aics, errors), columns=['ARMA(p,q) Model', 'AIC', 'RMSE'])
#         print(df)
        
        opt_idx = df[df.AIC==df.AIC.min()].index.tolist()
        print('Optimal model based on AIC is:')
        plt.figure(figsize=(20, 10))
        plt.plot(test, label='Expected')
        plt.plot(preds[opt_idx[0]], color='red', label='Predicted')
        model_name = df['ARMA(p,q) Model'][opt_idx[0]]
        plt.title(f'{model_name}; 33-Month Forecast; RMSE = %.3f; AIC = %.3f' % (df['RMSE'][opt_idx[0]], df.AIC.min()))
        plt.legend(loc='best')
        plt.show()
        print()
        
        opt_idx_rmse = df[df.RMSE==df.RMSE.min()].index.tolist()
        print('Optimal model based on RMSE is:')
        plt.figure(figsize=(20, 10))
        plt.plot(test, label='Expected')
        plt.plot(preds[opt_idx_rmse[0]], color='red', label='Predicted')
        model_name = df['ARMA(p,q) Model'][opt_idx_rmse[0]]
        plt.title(f'{model_name}; 33-Month Forecast; RMSE = %.3f; AIC = %.3f' % (df.RMSE.min(), df['AIC'][opt_idx_rmse[0]]))
        plt.legend(loc='best')
        plt.show()
    
    
    @staticmethod
    def ARIMA_model(data, maxp=3, maxq=3, maxd=1, *args, **kwargs):
        train, test = Stats.train_test_split(data)
        
        def_args_fit = {}
        def_args_fit['trend'] = kwargs.get('trend', 'nc')
        
        plags = range(maxp+1)
        dlags = range(maxd+1)
        qlags = range(maxq+1)
        p_d_q = [(x, z, y) for x in plags for z in dlags for y in qlags]
        all_models = {'ARIMA({0},{1},{2})'.format(x[0], x[1], x[2]): tsa.arima_model.ARIMA(train, x) for x in p_d_q}
        
        models = []
        preds = []
        errors = []
        aics = []
        
        for model in all_models:
            try:
                model_fit = all_models[model].fit(**def_args_fit)
                aic = model_fit.aic
                predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
                aics.append(aic)
                models.append(model)
                preds.append(predictions)
                error = mean_squared_error(test, predictions)
                errors.append(np.sqrt(error))
            except:
                pass
            
        df = pd.DataFrame(zip(models, aics, errors), columns=['ARIMA(p,d,q) Model', 'AIC', 'RMSE'])
#         print(df)
        
        opt_idx = df[df.AIC==df.AIC.min()].index.tolist()
        print('Optimal model based on AIC is:')
        plt.figure(figsize=(20, 10))
        plt.plot(test, label='Expected')
        plt.plot(preds[opt_idx[0]], color='red', label='Predicted')
        model_name = df['ARIMA(p,d,q) Model'][opt_idx[0]]
        plt.title(f'{model_name}; 33-Month Forecast; RMSE = %.3f; AIC = %.3f' % (df['RMSE'][opt_idx[0]], df.AIC.min()))
        plt.legend(loc='best')
        plt.show()
        print()
        
        opt_idx_rmse = df[df.RMSE==df.RMSE.min()].index.tolist()
        print('Optimal model based on RMSE is:')
        plt.figure(figsize=(20, 10))
        plt.plot(test, label='Expected')
        plt.plot(preds[opt_idx_rmse[0]], color='red', label='Predicted')
        model_name = df['ARIMA(p,d,q) Model'][opt_idx_rmse[0]]
        plt.title(f'{model_name}; 33-Month Forecast; RMSE = %.3f; AIC = %.3f' % (df.RMSE.min(), df['AIC'][opt_idx_rmse[0]]))
        plt.legend(loc='best')
        plt.show()
    
    
    @staticmethod
    def SARIMA_model(data, maxp=3, maxd=1, maxq=3, maxP=0, maxD=0, maxQ=0, maxs=[3, 6, 9, 12], *args, **kwargs):
        train, test = Stats.train_test_split(data)
        
        plags = range(maxp+1)
        dlags = range(maxd+1)
        qlags = range(maxq+1)
        p_d_q = [(x, z, y) for x in plags for z in dlags for y in qlags]
        
        Plags = range(maxP+1)
        Dlags = range(maxD+1)
        Qlags = range(maxQ+1)
        slags = range(maxs+1)
        P_D_Q_s = [(a, b, c, d) for a in Plags for b in Dlags for c in Qlags for d in slags]
        
        combs = [(x, y) for x in p_d_q for y in P_D_Q_s]
        
        all_models = {'SARIMA{0}{1}[{2}]'.format(comb[0], comb[1][:-1], comb[1][-1]): tsa.statespace.sarimax.SARIMAX(endog=train,
                                                                                                                     order=comb[0],
                                                                                                                     seasonal_order=comb[1]) 
                      for comb in combs}
        
        models = []
        preds = []
        errors = []
        aics = []
        
        for model in all_models:
            try:
                model_fit = all_models[model].fit()
                aic = model_fit.aic
                predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
                aics.append(aic)
                models.append(model)
                preds.append(predictions)
                error = mean_squared_error(test, predictions)
                errors.append(np.sqrt(error))
            except:
                pass
            
        df = pd.DataFrame(zip(models, aics, errors), columns=['SARIMA(p,d,q)(P,D,Q)[s] Model', 'AIC', 'RMSE'])
#         print(df)
        
        opt_idx = df[df.AIC==df.AIC.min()].index.tolist()
        print('Optimal model based on AIC is:')
        plt.figure(figsize=(20, 10))
        plt.plot(test, label='Expected')
        plt.plot(preds[opt_idx[0]], color='red', label='Predicted')
        model_name = df['SARIMA(p,d,q)(P,D,Q)[s] Model'][opt_idx[0]]
        plt.title(f'{model_name}; 33-Month Forecast; RMSE = %.3f; AIC = %.3f' % (df['RMSE'][opt_idx[0]], df.AIC.min()))
        plt.legend(loc='best')
        plt.show()
        
        opt_idx_rmse = df[df.RMSE==df.RMSE.min()].index.tolist()
        print('Optimal model based on RMSE is:')
        plt.figure(figsize=(20, 10))
        plt.plot(test, label='Expected')
        plt.plot(preds[opt_idx_rmse[0]], color='red', label='Predicted')
        model_name = df['SARIMA(p,d,q)(P,D,Q)[s] Model'][opt_idx_rmse[0]]
        plt.title(f'{model_name}; 33-Month Forecast; RMSE = %.3f; AIC = %.3f' % (df.RMSE.min(), df['AIC'][opt_idx_rmse[0]]))
        plt.legend(loc='best')
        plt.show()

