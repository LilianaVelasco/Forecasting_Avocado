
#!/usr/bin/python
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
#import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX


import warnings
warnings.filterwarnings("ignore")

SMALL_SIZE = 4
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

#Libraries to create the function check_error:
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_error

def check_error(orig, pred, name_col='', index_name=''):
    
    bias = np.mean(orig - pred)
    mse = mean_squared_error(orig, pred)
    rmse = sqrt(mean_squared_error(orig, pred))
    mae = mean_absolute_error(orig, pred)
    mape = np.mean(np.abs((orig - pred) / orig)) * 100
    
    error_group = [bias, mse, rmse, mae, mape]
    serie = pd.DataFrame(error_group, index=['BIAS','MSE','RMSE','MAE', 'MAPE'], columns=[name_col])
    serie.index.name = index_name
    
    return serie

# Function to plot data frame and plot errors to check residuals


import statsmodels.api as sm 
def plot_compare_error(data, nlagsp, figsize=(18,8)):
    
    # Creating the column error
    data['Error'] = data.iloc[:,0] - data.iloc[:,1]
    
    plt.figure(figsize=figsize)
    plt.xticks(rotation=90)
    ax1 = plt.subplot2grid((2,2), (0,0))
    ax2 = plt.subplot2grid((2,2), (0,1))
    ax3 = plt.subplot2grid((2,2), (1,0))
    ax4 = plt.subplot2grid((2,2), (1,1))
    
    #Plotting actual and predicted values
    ax1.plot(data.iloc[:,0:2])
    ax1.legend(['Real','Pred'])
    ax1.set_title('Real Value vs Prediction')
    
    # Error vs Predicted value
    ax2.scatter(data.iloc[:,1], data.iloc[:,2])
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residual')
    ax2.set_title('Residual vs Predicted Values')
    
    ## Residual QQ Plot
    sm.graphics.qqplot(data.iloc[:,2], line='r', ax=ax3)
    
    # Autocorrelation Plot of residual
    plot_acf(data.iloc[:,2], lags=nlagsp, zero=False, ax=ax4) #60, zero=False, ax=ax4)
    plt.tight_layout()
    plt.show()


#-----------------------------------
# Read/ Preparing the data
#-----------------------------------
value=sys.argv[1]
#print(str(sys.argv))

print(value)

#Read data

df = pd.read_csv (r'avocado.csv')
df = df[['Date',  'AveragePrice',  'Total Volume',  'type', 'region']]
df=df[df['region']=='TotalUS'] #TotalUS
#print (df)


#organic or conventional (but keep the name 'a_organic') 
if value=='organic':
    a_organic=df[df['type']=='organic']
elif value=='conventional':
    a_organic=df[df['type']=='conventional']

#print("Checking we have the right data")
print(a_organic)

#Setting index
a_organic=a_organic.sort_values(by=['Date'])
a_organic['Date']=pd.to_datetime(a_organic['Date'])
#a_organic=a_organic.set_index(['Date'], inplace=True)
a_organic.set_index('Date', inplace=True) 
#data_index_max=a_organic['Date'].max()


#__________________
#Split the data
#__________________

fraction=0.88
total_rows=len(a_organic['AveragePrice'].axes[0])
train_size=int(total_rows*fraction)
print(train_size)

train = a_organic.iloc[0:train_size,0:1].copy() 
train_cpy=train['AveragePrice'].copy() # A copy to plot together with model
test  = a_organic.iloc[train_size+1:total_rows,0:1].copy()

freq = pd.infer_freq(a_organic['AveragePrice'].axes[0])
print(freq)
series = pd.Series(train['AveragePrice'])#52 weeks
y=train['AveragePrice'].copy() 


#___________________________
#Training the model
#___________________________

#ARIMA(1, 1, 1) SARIMAX(1, 0, 1, 52)
model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,52), trend='n', enforce_stationarity=False, enforce_invertibility=False).fit()

print("________________________")
print("MODEL SUMMARY")
print(model.summary().tables[1])

# Nice way to check residuals follow a Gaussian distribution
model.plot_diagnostics(figsize=(15, 12))
plt.show()

train_pred = model.predict()
train_pred_cpy=train_pred.copy()
print(train_pred_cpy)
print(type(train_pred_cpy))
print(type(series))

cdf_index=a_organic[0:train_size].index
#print(cdf_index)
#print(type(cdf_index))


#________________________________________________
#Comparing the FIT with the trained data
#________________________________________________

#I need to create here a data frame from the series
compare_frame={'AveragePrice':series.reset_index(drop=True),'Predicted_AveragePrice':train_pred_cpy.reset_index(drop=True)}# the drop: starts all as columns
compare_df=pd.DataFrame(compare_frame)
#compare_df.set_index(cdf_index) #Does not want to change the index
#print(compare_df)

# Metrics to evaluate the model

error = series-train_pred_cpy
MFE = error.mean()
MAE = np.abs(error).mean()

#print(f'The error of each model value looks like this: {error}')
#print(f'The MFE error was {MFE}, the MAE error was {MAE}')

print(check_error(series,train_pred_cpy, 'Value','Training Base'))

#___________________________________________________________
#compare_df contains trained and fitted 
#___________________________________________________________

plot_compare_error(compare_df,60)  


#___________________________________________________________
# Now forecast in the test data:all at the same time for
# a rough idea
#___________________________________________________________

predictions=model.predict(
    start=len(train), end=len(train)+len(test)-1,
    dynamic=False)

# create a comparison dataframe
compare_test_frame={'AveragePrice':test['AveragePrice'].reset_index(drop=True),'Predicted_AveragePrice':predictions.reset_index(drop=True)}
compare_test_df=pd.DataFrame(compare_test_frame)

print(compare_test_df)


#print('PREDICTIONS WITH THE MODEL')
#print(predictions)

#creating the basis of error in the test
error_test = check_error(compare_test_df['AveragePrice'], 
                        compare_test_df['Predicted_AveragePrice'], 
                        name_col='Value Comp. Pred.vs. Fit',
                       index_name='Testing Base')


print(' TEST and PREDICTION')
plot_compare_error(compare_test_df,len(compare_test_df)-1)
print(error_test)

#dti = pd.date_range(data_index_max, periods=5, freq='W-SUN')
print("____________________________")
print("Forecast for one period")
print(model.forecast()[0])
#print("on")
#print(  dti[1] )
nstepsfor=int(15)
pred_uc=model.forecast(steps=nstepsfor)[0]

#print(pred_ci = pred_uc.conf_int())

print("CONFIDENCE INTERVALS")
print("____________________________")
print("Forecast for")
print(nstepsfor)

#for t in range(0,nstepsfor):
#    print(pred_uc[t])

#plt.plot(pred_uc)
#plt.show()
    
#print(pred_uc)


#_________________________
#Getting the coefficients of the fit
#_________________________

print('Model parameters')
print(model.params)

