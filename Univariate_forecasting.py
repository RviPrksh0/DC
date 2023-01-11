import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from prophet import Prophet
from prophet.make_holidays import make_holidays_df
import copy
sns_c = sns.color_palette(palette='deep')

from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet.diagnostics import performance_metrics
from statsmodels.tsa.seasonal import seasonal_decompose
holidays = make_holidays_df(year_list=year_list, country='UK')

def create_train_test_frame_(dff , threshold = '01/07/2012',parameter='price'):
    df=copy.deepcopy(dff)
    df.rename(columns={'date':'ds',parameter:'y'},inplace=True)
    threshold_date = pd.to_datetime(threshold)
    df['ds'] = pd.to_datetime(df['ds']).astype('datetime64[ns]')
    mask = df.reset_index(drop=True)['ds'] < threshold_date
    # Split the data and select `ds` and `y` columns.
    df_train = df.reset_index(drop=True)[mask][['ds', 'y']]
    df_test = df.reset_index(drop=True)[~ mask][['ds', 'y']]
    dx = df.reset_index(drop=True)[['ds', 'y']]
    return df_train, df_test, threshold_date, dx
#df_train, df_test, threshold_date, dx = create_train_test_frame_(dff,parameter='price')

def test_train_visualize(df_train, df_test):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    sns.lineplot(x='ds', y='y', label='y_train', data=df_train, ax=ax)
    sns.lineplot(x='ds', y='y', label='y_test', data=df_test, ax=ax)
    ax.axvline(threshold_date, color=sns_c[3], linestyle='--', label='train test split')
    ax.legend(loc='upper left')
    ax.set(title='Dependent Variable', ylabel='');
#test_train_visualize(df_train,df_test)

def build_model():
    """Define forecasting model."""
    # Create holidays data frame. 
    
    model = Prophet(
         yearly_seasonality=True,
         weekly_seasonality=True,
         daily_seasonality=True, 
        holidays = holidays,
        seasonality_mode='additive',
        interval_width=0.95, 
        mcmc_samples = 0,
        changepoint_prior_scale=0.05,seasonality_prior_scale=4,n_changepoints=2,
    )

    model.add_seasonality(
        name='monthly', 
        period=30.5, 
        fourier_order=6,mode='additive',
    )
    
    return model
#model_project = build_model()

def project_visualize(model = model_project, data=dx, threshold_date = threshold_date,train= df_train, test = df_test,forecast_length=170):

    threshold_test=df_test['ds'].iloc[-1]
  ## make model
    model.fit(data)
    future = model.make_future_dataframe(periods=forecast_length, freq='D')
    # Generate predictions. 
    forecast = model.predict(df=future)
    ### Select a few features
    test2 = forecast[['ds','yhat_lower', 'yhat_upper','yhat','trend']]#[:169]
    test2['y']= data['y']

    predict_mask=forecast['ds']<=threshold_test
    predict= forecast[predict_mask]
    

    mask2 = predict['ds'] < threshold_date
    forecast_test = predict[~ mask2]
    forecast_train=predict[mask2]
    
    

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    ax.fill_between(
      x=forecast['ds'],
      y1=forecast['yhat_lower'],
      y2=forecast['yhat_upper'],
      color=sns_c[2], 
      alpha=0.25,
      label=r'0.95 credible_interval'
    )

    sns.lineplot(x='ds', y='y', label='y_train', data=train, ax=ax)
    sns.lineplot(x='ds', y='y', label='y_test', data=test, ax=ax)
    sns.lineplot(x='ds', y='yhat', label='y_hat', data=forecast, ax=ax)
    ax.axvline(threshold_date, color=sns_c[3], linestyle='--', label='train test split')
    ax.legend(loc='upper left')
    ax.set(title='Dependent Variable', ylabel='');

    return forecast, test2, forecast_train, forecast_test
#forecast, test2, forecast_train, forecast_test = project_visualize(model = model_project, data = dx, threshold_date = threshold_date,train= df_train, test = df_test)

def error_analysis_back_testing(nx = 103, df_train = df_train, df_test = df_test, forecast_train = forecast_train, forecast_test = forecast_test, forecast = forecast, model = model_project):
    n = nx
    print('r2 train: {}'.format(r2_score(y_true=df_train['y'], y_pred=forecast_train['yhat'])))
    print('r2 test: {}'.format(r2_score(y_true=df_test['y'], y_pred=forecast_test['yhat'])))
    print('---'*10)
    print('mae train: {}'.format(mean_absolute_error(y_true=df_train['y'], y_pred=forecast_train['yhat'])))
    print('mae test: {}'.format(mean_absolute_error(y_true=df_test['y'], y_pred=forecast_test['yhat'])))
    print('training mape: ',(mean_absolute_error(y_true=df_train['y'], y_pred=forecast_train['yhat'])/df_train['y'].mean())*100)
    print('testing mape: ',(mean_absolute_error(y_true=df_test['y'], y_pred=forecast_test['yhat'])/df_train['y'].mean())*100)
    ### Error analysis
    forecast_test.loc[:, 'errors'] = forecast_test.loc[:, 'yhat'] - df_test.loc[:, 'y']

    errors_mean = forecast_test['errors'].mean()
    errors_std = forecast_test['errors'].std()

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
 
    sns.distplot(a=forecast_test['errors'], ax=ax, bins=15, rug=True)
    ax.axvline(x=errors_mean, color=sns_c[2], linestyle='--', label=r'$\mu$')
    ax.axvline(x=errors_mean + 2*errors_std, color=sns_c[3], linestyle='--', label=r'$\mu \pm 2\sigma$')
    ax.axvline(x=errors_mean - 2*errors_std, color=sns_c[3], linestyle='--')
    ax.legend()
    ax.set(title='Model Errors (Test Set)');

    ### Auto Correlation Check
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    sns.scatterplot(x='index', y='errors', data=forecast_test.reset_index(), ax=ax)
    ax.axhline(y=errors_mean, color=sns_c[2], linestyle='--', label=r'$\mu$ ')
    ax.axhline(y=errors_mean + 2*errors_std, color=sns_c[3], linestyle='--', label=r'$\mu \pm 2\sigma$')
    ax.axhline(y=errors_mean - 2*errors_std, color=sns_c[3], linestyle='--')
    ax.legend()
    ax.set(title='Autocorrelation (Test Set)');


    ### PACF and ACF check 

    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    plot_acf(x=forecast_test['errors'], ax=ax[0])
    plot_pacf(x=forecast_test['errors'], ax=ax[1]);


    #### Trend fit check
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    #sns.lineplot(x='ds', y='trend', data=df, label='trend_true', ax=ax)

    ax.fill_between(
      x=forecast['ds'],
      y1=forecast['trend_lower'],
      y2=forecast['trend_upper'],
      color=sns_c[1], 
      alpha=0.25,
      label=r'0.95 credible_interval'
    )

    sns.lineplot(x='ds', y='trend', data=forecast, label='trend_fit', ax=ax)
    ax.legend(loc='upper left')
    ax.set(title='Trend Fit', ylabel='');
  

    ### Model Diagnostic
    from prophet.diagnostics import cross_validation

    df_cv = cross_validation(
      model=model, 
      initial='365 days', 
      period='30 days', 
      horizon = '180 days'
    )
    df_p = performance_metrics(df=df_cv, rolling_window=0.1)
    fig = plot_cross_validation_metric(df_cv=df_cv, metric='mape', rolling_window=0.1, figsize=(15, 10))
    return None
#error_analysis_back_testing(nx = 62, df_train = df_train, df_test = df_test, forecast_train = forecast_train, forecast_test = forecast_test, forecast = forecast)

def univariate_forecasting(data,forecast_length,save_path,parameter):
    
    '''
    df: It should have dates in a column named ds and required parameter as y
    threshold: string: Sould be in format 'YYYY-MM-DD', that diffrentiates train from test
    forecast_lenght: int
    save_path:
    parameter: parameter for which univariate forecasting is done

    Following function does step modelling for price and for the rest uses prophet for univariate prediction and saves the file in given location.
    '''
    df=copy.deepcopy(data)
    threshold_id=len(df)//4
    threshold=str(df['date'][threshold_id])

    df=df[['date',parameter]]

    if parameter=='price' or parameter=='Price':
        df_price=copy.deepcopy(df)
        df_price.set_index('date',inplace=True)
        df_price.index.freq='D'

        result_add = seasonal_decompose(df_price[parameter], model='additive', extrapolate_trend='freq')
        result_mul = seasonal_decompose(df_price[parameter], model='multiplicative', extrapolate_trend='freq')

        plt.rcParams.update({'figure.figsize': (15,10)})
        result_add.plot().suptitle('Additive decomposition', x=0.2, fontweight='bold')
        plt.show()

        result_mul.plot()
        plt.show()

        df_reconstructed = pd.concat(
            [result_mul.seasonal, 
            result_mul.trend, 
            result_mul.resid, 
            result_mul.trend * result_mul.resid, 
            result_mul.observed], axis=1)
        df_reconstructed.columns = ['seasonal', 'trend', 'remainders', 'seasonal_adj', 'actual_values']

        df_forecast = df_reconstructed.iloc[-1*forecast_length:,:]
        df_forecast.index.freq='D'
        df_forecast = df_forecast.set_index(df_forecast.index.shift(forecast_length))
        df_forecast = df_forecast.drop('actual_values', axis=1)
        df_forecast[['trend', 'remainders', 'seasonal_adj']] = np.nan

        df_forecast['trend'] = df_reconstructed.loc[df_reconstructed.index[-1], 'trend']
        df_forecast['remainders'] = df_reconstructed.loc[df_reconstructed.index[-1], 'remainders']
        df_forecast['seasonal_adj'] = df_forecast['trend'] * df_forecast['remainders'] 
        df_forecast['forecast'] = df_forecast['seasonal_adj'] * df_forecast['seasonal']
        df_plot = pd.concat([df_reconstructed, df_forecast], axis=0)
        plt.rcParams.update({'figure.figsize': (20,7)})
        df_plot[['actual_values', 'forecast']].plot()
        df_plot.iloc[-50:,:][['actual_values', 'forecast']].plot()
        plt.show()

        df_plot['seas_trend'] = df_plot['seasonal']*df_plot['trend']*df_plot['remainders']

        df_plot.to_csv(save_path+f'Univariate forecast {parameter}.csv')
    
    else:
        ## Creating Train test dataset
        df_train, df_test, threshold_date, dx = create_train_test_frame_(df, threshold,parameter )
        test_train_visualize(df_train, df_test)
        
        model_project = build_model()
        
        ### Visualizing Results ######
        forecast, test2, forecast_train, forecast_test = project_visualize(model = model_project, data = dx, threshold_date = threshold_date,train= df_train, test = df_test,forecast_length=forecast_length)
        
        ## Analysing Result
        error_analysis_back_testing(nx = 62, df_train = df_train, df_test = df_test, forecast_train = forecast_train, forecast_test = forecast_test, forecast = forecast)
        test2.to_csv(save_path+f'Univariate forecast {parameter}.csv')
    return forecast, test2, forecast_train, forecast_test
    
    ## Now Forecasting