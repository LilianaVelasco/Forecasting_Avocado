#!/usr/bin/python
import sys
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm


#__________________________
#  Read/Prepare data
#__________________________

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

#print("Checking we have the right data")
print(a_organic)

#Sort/Setting index
a_organic=a_organic.sort_values(by=['Date'])
a_organic['Date']=pd.to_datetime(a_organic['Date'])
#a_organic=a_organic.set_index(['Date'], inplace=True)
#a_organic.set_index('Date', inplace=True)
print (a_organic)
ndf = pd.DataFrame(a_organic, columns=['Date','AveragePrice']).set_index('Date')
print (ndf)

fraction=0.88
total_rows=len(ndf)
train_size=int(total_rows*fraction)
train = ndf.iloc[:train_size, :]
test =  ndf.iloc[train_size:, :]

#print(test)
#print("Lenght of test")
#print(len(test))

#Also works with sm.tsa.SARIMAX
sarima_model =SARIMAX(train, order=(1, 1, 0), seasonal_order=(0, 1, 1, 52), enforce_invertibility=False, enforce_stationarity=False)
sarima_fit = sarima_model.fit()
sarima_forecast = sarima_fit.get_forecast(35,dynamic=True)
#The dynamic=False argument ensures that we produce one-step ahead forecasts, meaning that forecasts at each point are generated using the full history up to that point.
#Get confidence intervals
predicted_intervals = sarima_forecast.conf_int(alpha=0.05)

print("Central values of the forecast")
print(sarima_forecast)
print("Predicted intervals, four periods")
print(predicted_intervals)


ax = ndf.plot(label='Observed', figsize=(20, 10))
sarima_forecast.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(predicted_intervals.index,
                predicted_intervals.iloc[:, 0],
                predicted_intervals.iloc[:, 1], color='k', alpha=.25)
ax.set_title('Forecast for Avocado prices')
ax.set_xlabel('Date')
ax.set_ylabel('Average Price')

plt.legend()
plt.show()
