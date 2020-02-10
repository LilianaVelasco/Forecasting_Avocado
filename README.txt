Type
python3 file.py conventional
or
python3 file.py	organic

to select the different types, in each of the following programs

The order in which the programs should be run is as follows

1. gather_prepare_data.py :
		       Prepares the data and makes a scatter plot and a correlation plot to check for general behaviour 


2. check_stationarity_pacf_acf.py


3. choice_of_model_ap.py :
		      Decides on correlation, check for stationarity with the Adjusted Dickey-Fuller test and the  KPSS tests. Checks for seasonality and trend, checking if the series after the first derivation is more stationary than the original series.

4. determine_best_ARIMA_SARIMA.py:  Using BIC and AIC tests confirms the chosen ARIMA parameters or selects a better one.

5. evaluate_error_models_validation.py

6.   compare_validate_forecast_ARIMA.py:   Here we compare and validate the forecast,  we use 
a  rolling forecast to account for the dependence on observations in prior time steps for differencing (in case it is needed) and the AR model. We perform this rolling forecast by recreating the ARIMA model after each new observation is received.

We manually keep track of all observations in a list called history that is seeded with the training data and to which new observations are appended each iteration.








# Forecasting_Avocado
