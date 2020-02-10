import pandas as pd
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import concat
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
#import seaborn as sns
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

SMALL_SIZE = 4
MEDIUM_SIZE = 10
BIGGER_SIZE = 12


#Prepare data

df = pd.read_csv (r'avocado.csv')
df = df[['Date',  'AveragePrice',  'Total Volume',  'type', 'region']]
#print (df)



#organic: Select the organic type
df=df[df['region']=='TotalUS'] #TotalUS
#a_organic=df[df['type']=='organic']
a_organic=df[df['type']=='conventional']

a_organic=a_organic.sort_values(by=['Date'])
print (a_organic)
a_organic.set_index('Date', inplace=True)

#Define series
freq = pd.infer_freq(a_organic.axes[0])
#print(freq)
series = pd.Series(a_organic['AveragePrice']) #complete series here


#lag_plot(a_organic['AveragePrice']).plot(figsize=(15,4))
#plt.show()

#values = DataFrame(series.values)
#dataframe = concat([values.shift(1), values], axis=1)
#dataframe.columns = ['t-1', 't+1']
#result = dataframe.corr()
#print(result)

#autocorrelation_plot(series)
#plt.show()


#Split the data

total_rows=len(a_organic.axes[0])
train_size=int(total_rows*.89)
print(train_size)

train = a_organic.iloc[0:train_size,0:1].copy() #iloc: accesses firs the rows, then columns
train_cpy=train['AveragePrice'].copy() # A copy to plot together with model

test  = a_organic.iloc[train_size+1:total_rows,0:1].copy()

#plt.plot(train['AveragePrice'], color='b')
#plt.plot(test['AveragePrice'], color='orange')
#plt.legend(['Train','Test'])
#plt.xticks(rotation=90)
#plt.rc('xtick', labelsize=SMALL_SIZE)  
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=SMALL_SIZE)
#plt.xlabel('Date', fontsize=SMALL_SIZE)
#plt.ylabel('Price',fontsize=MEDIUM_SIZE)
#plt.show()



#___________________________________
#creating a function to plot the graph and show the test result:
def adfuller_test(serie, figsize=(18,4), plot=True, title=""):
    if plot:
        serie.plot(figsize=figsize, title=title)
        plt.show()
    #Dickey Fuller test on the first differentiation
    adf = adfuller(serie)
    output = pd.Series(adf[0:4], index=['Dickey Fuller Statistical Test', 'P-value',
                                        'Used Lags', 'Number of comments used'])
    output = round(output,4)
    
    for key, value in adf[4].items():
        output["Critical Value (%s)"%key] = value.round(4) 
    return output


#adfuller_test(train['AveragePrice'].diff().dropna(), title='Prices with first differentiation')
#_________________________
# Deciding model 
#_________________________

# evaluate an ARIMA model for a given order (p,d,q)

def evaluate_arima_model(X, arima_order,seasonar_order):
    # prepare training dataset

    train_size = int(len(X) * 0.88)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
#        model = ARIMA(history, order=arima_order)
        model =SARIMAX(history, order=arima_order,seasonal_order=seasonar_order, trend='n')
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

def evaluate_models(dataset, p_values, d_values, q_values, P_svalues, D_svalues, Q_svalues):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                for P in P_svalues:
                    for D in D_svalues:
                        for Q in Q_svalues:
                            seasonar_order=(P,D,Q,52)
                            try:
                                mse = evaluate_arima_model(dataset, order,seasonar_order)
                                if mse < best_score:
                                    best_score, best_cfg = mse, order
                                print('ARIMA%s MSE=%.3f' % (order,mse))
                            except:
                                continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

# evaluate parameters
p_values = [0, 1]#[0, 1, 2, 4, 6, 8, 10]
d_values = [0, 1]
q_values = [0, 1]#range(0,3)
P_svalues= [0, 1]
D_svalues= [0, 1]
Q_svalues= [0, 1]
 
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values,P_svalues, D_svalues, Q_svalues)
    
