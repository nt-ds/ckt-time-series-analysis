#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports

import json
import quandl


# In[2]:


class Data():
    
    with open("quandl_secret.json") as f:
        d = json.load(f)
        __API_KEY = d["API Key"]
    quandl.ApiConfig.api_key = __API_KEY
    
    @classmethod
    def get_commod_data(cls, *args, **kwargs):
        data = {'oil': cls._get_oil_data, 'gasoline': cls._get_gasoline_data,
                'natgas': cls._get_natgas_data, 'corn': cls._get_corn_data,
                'wheat': cls._get_wheat_data}
        results = {x: data[x](*args, **kwargs) for x in data.keys()}
        return results
    
    @classmethod
    def get_econ_data(cls, *args, **kwargs):
        data = {'cons_sent': cls._get_mich_cons_sent_data, 'fred_gdp': cls._get_fred_gdp_data,
                'us_yields': cls._get_us_yields_data}
        results = {x: data[x](*args, **kwargs) for x in data.keys()}
        return results
    
    #commodities data
    @staticmethod
    def _get_oil_data(*args, **kwargs):
        data = quandl.get('CHRIS/CME_QM1')
        return data
    @staticmethod
    def _get_gasoline_data(*args, **kwargs):
        data = quandl.get('CHRIS/ICE_N1')
        return data
    @staticmethod
    def _get_natgas_data(*args, **kwargs):
        data = quandl.get('CHRIS/CME_NG2')
    @staticmethod
    def _get_corn_data(*args, **kwargs):
        data = quandl.get('CHRIS/LIFFE_EMA2')
        return data
    @staticmethod
    def _get_wheat_data(*args, **kwargs):
        data = quandl.get('CHRIS/CME_W4')
        return data
    
    # Economic data
    @staticmethod
    def _get_mich_cons_sent_data(*args, **kwargs):
        data = quandl.get('UMICH/SOC1')
        return data
    @staticmethod
    def _get_fred_gdp_data(*args, **kwargs):
        data = quandl.get('FRED/GDP')
        return data
    # interest rate data
    @staticmethod
    def _get_us_yields_data(*args, **kwargs):
        data = quandl.get('USTREASURY/YIELD')
        return data

