#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as smapi
import statsmodels.tsa as tsa
import warnings

sns.set_style("whitegrid")
warnings.filterwarnings("ignore")


# In[2]:


class Stats():
    
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
        def_args['autolag'] = kwargs.get('autolag', 'AIC')
        def_args['store'] = kwargs.get('store', False)
        def_args['regresults'] = kwargs.get('regresults', False)
        
        result = tsa.stattools.adfuller(data.values.reshape(-1,), **def_args)
        print('ADF Statistic: %f' % result[0])
        print('p-value:       %f' % result[1])
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
    
    
    @staticmethod
    def compute_acf(data,*args,**kwargs):
        # return the sample autocorrelation function
        # by default return the confidence intervals
    
        def_args = {}
        def_args['unbiased']=kwargs.get('unbiased',False)
        def_args['nlags']=kwargs.get('nlags',36)
        def_args['qstat']=kwargs.get('qstat',False)
        def_args['alpha']=kwargs.get('alpha',0.05)
        
        results = tsa.stattools.acf(data,**def_args)
        return results
    
    @staticmethod
    def compute_pacf(data,*args,**kwargs):
        # return the sample autocorrelation function
        # by default return the confidence intervals
        def_args = {}
        def_args['nlags']=kwargs.get('nlags',36)
        def_args['method']=kwargs.get('method','ywunbiased')
        def_args['alpha']=kwargs.get('alpha',0.05)
        
        results = tsa.stattools.pacf(data,**def_args)
        return results
    
    @staticmethod
    def AR_model(data,*args,**kwargs):
        def_args = {}
        def_args['dates'] = kwargs.get('dates',None)
        def_args['freq'] = kwargs.get('freq',None)
        def_args['missing'] = kwargs.get('missing',None)
        model= tsa.ar_model.AR(data.values.reshape(-1,),**def_args)
        return model
                
    @staticmethod
    def ARMA_model(data,maxp=3,maxq=3,*args,**kwargs):
        plags=range(maxp+1)
        qlags=range(maxq+1)
        p_q = [(x,y) for x in plags for y in qlags]
        all_models = {'arma({0},{1})'.format(x[0],x[1]):tsa.arima_model.ARMA(data,x,) for x in p_q}
        return all_models
        
        
    @staticmethod
    def ARIMA_model(data,maxp=3,maxq=3,maxd=1,*args,**kwargs):
        plags=range(maxp+1)
        dlags=range(maxd+1)
        qlags=range(maxq+1)
        p_d_q = [(x,z,y) for x in plags for z in dlags for y in qlags]
        all_models = {'arima({0},{1},{2})'.format(x[0],x[1],x[2]):tsa.arima_model.ARIMA(endog=data,order=x,) for x in p_d_q}
        return all_models
        
    @staticmethod
    def SARIMA_model(data,maxp=3,maxq=3,maxd=1,maxP=0,maxD=0,maxQ=0,maxs=0,*args,**kwargs):
        plags=range(maxp+1)
        dlags=range(maxd+1)
        qlags=range(maxq+1)
        p_d_q = [(x,z,y) for x in plags for z in dlags for y in qlags]
        
        Plags=range(maxP+1)
        Dlags=range(maxD+1)
        Qlags=range(maxQ+1)
        slags=range(maxs+1)
        P_D_Q_s = [(a,b,c,d) for a in Plags for b in Dlags for c in Qlags for d in slags]
        
        all_models = {'sarima(({0},{1},{2})X({3},{4},{5},{6})'.format(x[0],x[1],x[2]):tsa.arima_model.SARIMAX(data,None,
                                                                                                              order=x,seasonal_order=y) 
                      for x in p_d_q for y in P_D_Q_s}

        
    
