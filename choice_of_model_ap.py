#!/usr/bin/python
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import kpss


import warnings

SMALL_SIZE = 4
MEDIUM_SIZE = 10
BIGGER_SIZE = 12


warnings.filterwarnings("ignore")

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

#print(a_organic)
#Sort data


a_organic=a_organic.sort_values(by=['Date'])
a_organic.set_index('Date', inplace=True)

#Checking we have the right type
print(a_organic)

#-----------------------------------
#Looking for autocorrelation
#-----------------------------------


#This one way
#plt.figure(figsize=(20,10))
#plt.title('Autocorrelation', fontsize=40)
#ax=plt.axes()
#ax.xaxis.set_major_locator(plt.MaxNLocator(10))
#plt.ylabel('Autocorrelation', fontsize= 30)
#plt.xlabel('y(t)',fontsize= 30)
#autocorrelation_plot(series) #Calculates automatically 95% CL
#plt.show()

#This is another way

y=a_organic['AveragePrice'].copy() 


def corr_plot_attributes(arg):
    ax=plt.axes()
    plt.xticks(fontsize= 30)
    plt.xlabel('Number of Lags',fontsize= 30)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.yticks(fontsize= 30)
    plt.ylabel(arg, fontsize= 30)
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))
    return

#Calculates by hand  95% CL
plt.figure(figsize=(15,8))
plt.title('AUTO-CORRELATION PLOT (ACF)', fontsize=40)
lag_acf= acf(y, nlags=54) #There are 134 weeks here period=lag
# nlags: we can leave it out and is calculated
type_corr="Correlation"
corr_plot_attributes(str(type_corr))
plt.axhline(y=-1.96/np.sqrt(len(y)),linestyle='--',color='goldenrod')
plt.axhline(y=1.96/np.sqrt(len(y)),linestyle='--',color='goldenrod')
plt.plot(lag_acf, marker="o")
plt.show()

plt.figure(figsize=(15,8))
plt.title('PARTIAL AUTO-CORRELATION PLOT (PACF)', fontsize=40)
lag_pacf= pacf(y, nlags=54)
type_corr="Auto Correlation"
corr_plot_attributes(str(type_corr))
plt.axhline(y=-1.96/np.sqrt(len(y)),linestyle='--',color='goldenrod')
plt.axhline(y=1.96/np.sqrt(len(y)),linestyle='--',color='goldenrod')
plt.plot(lag_pacf, marker="o")
plt.show()



#------------------------------------------------------
#Defining a series to decompose trend and seasonality
#------------------------------------------------------

#Every time series can be broken down into 3 parts: trend, seasonality and residuals, which is what remains after removing the first two parts from the series, below the separation of these parts.

freq = pd.infer_freq(a_organic.axes[0])
print(freq)
series = pd.Series(a_organic['AveragePrice'])

# weekly chart with cycles that repeat every 52 weeks (1 year)
result = seasonal_decompose(series, freq=52, model='additive')


fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(15,8))
#plt.title('Decomposition', fontsize=40)
result.trend.plot(ax=ax1)
result.resid.plot(ax=ax2)
result.seasonal.plot(ax=ax3)
#Complete plot
result.plot()
plt.xticks(fontsize= 4, rotation=60)
plt.show()

#______________________________________________________________
#Checking Gaussian distribution of prices
#______________________________________________________________
plt.title('Does the price follows a Gaussian distribution?', fontsize=15)
plt.hist(series, bins=55)
plt.show()

# identify outliers with boxes
red_square = dict(markerfacecolor='r', marker='s')
fig5, ax5 = plt.subplots()
ax5.set_title('Horizontal Boxes')
ax5.boxplot(a_organic['AveragePrice'], vert=False, flierprops=red_square)
plt.show()

# identify outliers with standard deviation

# calculate summary statistics
data_mean=np.mean(a_organic['AveragePrice'])
print('--------------------------------------')
print('MEAN AND AVERAGE')
print('--------------------------------------')
print('MEAN OF AVERAGE AVOCADO PRICES')

print(data_mean)
data_std=np.std(a_organic['AveragePrice'])
print('STANDARD DEVIATION OF AVERAGE AVOCADO PRICES')
print(data_std)
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off

# identify outliers: Treatment as Gaussian

y=a_organic['AveragePrice'].copy() 
outliers = [x for x in y if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))
# remove outliers
outliers_removed = [x for x in y if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))



#______________________________________________________________
#Checking for a stationary series
#______________________________________________________________
print('--------------------------------------')
print('STATISTICAL TEST OF STATIONARITY')
print('--------------------------------------')
#Dickey Fuller Test
print('Dickey Fuller Test')
print(' ')
#series.asfreq(freq) #Here assigns the weekly frequency
adf_test=  adfuller(series)
print("ADF = " + str(adf_test[0]))
print("p-value = " +str(adf_test[1]))
print("Used Lags = " +str(adf_test[2]))
print("Critical values = " +str(adf_test[4]))

# KPSS tests
def kpss_test(series, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')
    
print('KPSS Test')    
print('  ')
kpss_test(series)

