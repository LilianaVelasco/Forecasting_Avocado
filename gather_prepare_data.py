#!/usr/bin/python
import sys
import pandas as pd
import pandas as pd
from pandas import DataFrame
from pandas import concat
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import warnings

#plt.style.use('classic')

warnings.filterwarnings("ignore")

value=sys.argv[1]
#print(str(sys.argv))

print(value)

#Read data

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
#Sort data
a_organic=a_organic.sort_values(by=['Date'])
a_organic['Date']=pd.to_datetime(a_organic['Date'])
#a_organic=a_organic.set_index(['Date'], inplace=True)
#a_organic.set_index('Date', inplace=True) #This works also

#Checking we have the right type
print(a_organic)

#Scatter plot
plt.figure(figsize=(20,15))
plt.title('Organic Avocado Prices', fontsize=30)
plt.scatter(a_organic['Date'],a_organic['AveragePrice'])
ax=plt.axes()
plt.xticks(fontsize= 30, rotation=25)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
plt.yticks(fontsize= 30)
ax.yaxis.set_major_locator(plt.MaxNLocator(10))
plt.show()

#Look for the correlation

series = pd.Series(a_organic['AveragePrice']) 

plt.figure(figsize=(20,10))
plt.title('Price Correlation', fontsize=40)
lag_plot(a_organic['AveragePrice'])
ax=plt.axes()
plt.xticks(fontsize= 30)
plt.xlabel('y(t)',fontsize= 30)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
plt.yticks(fontsize= 30)
plt.ylabel('y(t+1)', fontsize= 30)
ax.yaxis.set_major_locator(plt.MaxNLocator(10))
plt.show()

#We also check how correlated are the variables
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print("Checking correlation coefficients")
print(result)




