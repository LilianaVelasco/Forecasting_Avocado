#Using BIC and AIC tests confirms the chosen ARIMA parameters or selects a better one.
#!/usr/bin/python
import sys
import pandas as pd
import numpy as np

import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import statsmodels.stats as sms
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


#Read data

value=sys.argv[1]
#print(str(sys.argv))

print(value)


df = pd.read_csv (r'avocado.csv')
df = df[['Date',  'AveragePrice',  'Total Volume',  'type', 'region']]
df=df[df['region']=='TotalUS'] #TotalUS
#print (df)



#Prepare data
#organic or conventional (but keep the name 'a_organic') 
if value=='organic':
    a_organic=df[df['type']=='organic']
elif value=='conventional':
    a_organic=df[df['type']=='conventional']

print(a_organic)


a_organic=a_organic[['Date',  'AveragePrice', 'region']].reset_index(drop=True)
a_organic['Date']=pd.to_datetime(a_organic['Date'])
a_organic.set_index('Date', inplace=True)

#Selects train series
split_factor=0.88
total_rows=len(a_organic['AveragePrice'].axes[0])
train_size=int(total_rows*split_factor)
validation_size=total_rows-train_size
train  = a_organic.iloc[0:train_size,0:1].copy()
valid  = a_organic.iloc[train_size+1:total_rows,0:1].copy()
print(a_organic)
series = pd.Series(train['AveragePrice'])

#Implements BIC AND AIC tests

# pick best order by aic then check bic
# smallest aic value wins
best_aic = np.inf 
best_order = None
best_seasonal_o = None
best_mdl = None

rngAR   = range(0,2)
rngI    = range(0,2)
rngMA   = range(0,2)
rngSAR  = range(0,2)
rngIS   = range(0,2)
rngSMA  = range(0,2)


# SARIMAX(series, order=(1,0,1), seasonal_order=(1,1,1,52), trend='n', enforce_stationarity=False, enforce_invertibility=False).fit()

for i in rngAR:
    for j in rngI:
        for k in rngMA:
            for P in rngSAR:
                for Q in rngSMA:
                    for D in rngIS:
                        try:
                            tmp_mdl = SARIMAX(series,
#                                            order=(i, j, k),seasonal_order=(P,D,Q,52), trend='n', enforce_stationarity=False, enforce_invertibility=False).fit() #(method='mle',trend='nc')
                                            order=(i, j, k),seasonal_order=(P,D,Q,52), trend='n').fit() 
                            tmp_aic = tmp_mdl.aic
                            if  tmp_aic <  best_aic:
                                best_aic = tmp_aic
                                best_order = (i, j, k)
                                best_seasonal_o= (P, D, Q, 52)
                                best_mdl = tmp_mdl
                        except: continue


print('Best AIC: %6.2f | order: %s | sorder: %s ' %(best_aic, best_order,best_seasonal_o))
#print(best_mdl.params)


#___________________________________
#Measures of performance
#___________________________________



#JungBox test
# A value of p must be greater than 0.05 which states that the residuals are independent at the 95% level and thus the 'best_mdl' provides a good model fit

print('Here the value of p must be greater than 0.05')
sms.diagnostic.acorr_ljungbox(best_mdl.resid, lags=[20], boxpierce=False)


# Check if residuals are noise or not
from statsmodels.stats.stattools import jarque_bera

score, pvalue, _, _ = jarque_bera(best_mdl.resid)

print('score= ', score)
print('pvalue= ', pvalue)
if pvalue < 0.10:
    print ('The residuals may not be normally distributed.')
else:
    print ('The residuals seem normally distributed.')

max_lag=train_size-3

print(best_mdl.summary())

