import pandas as pd
import numpy as np
import os
import copy
from datetime import date
import holidays

gb_holidays = holidays.GB()

def loading_dataframe(filename_list,save_path='',SKU_codes=['6018098','7786645','7786652','7916386','7916395','7970839','7989660']):
    req_df=pd.DataFrame()
    
    for filename in filename_list:
        if filename=='FullSales':
            df = pd.read_csv("FullSales.csv",encoding='iso-8859-1') #encoding='iso-8859-1'
            df = df.drop(['Unnamed: 3'], axis=1)
            df=df.iloc[1:]
            df.reset_index(inplace=True,drop=True)
            df.drop('Date',inplace=True,axis=1)
            df1 = df.set_index(['Store', 'Unnamed: 1', 'Item']).fillna(0)
            year_list=list(df1.columns)
            df2 = pd.melt(df1, value_vars=year_list , var_name='date',value_name='sold_value', ignore_index=False,).reset_index()
            df3 = df2[[ 'date', 'Store','Item','sold_value']]
            df3.columns = [ 'date', 'store','item','value']
            df3['holiday'] = df3['date'].apply(lambda m : gb_holidays.get(m))
            df3['holiday'] = df3['holiday'].fillna('normal')
            df3['date'] = pd.to_datetime(df3['date'], format = '%d/%m/%Y')
            data = df3#.fillna(0)
            s = pd.to_numeric(data['value'], errors='coerce')
            data['value'] = s
        
        elif filename=='NewSales':
            df = pd.read_csv(filename+".csv")
            df = df.drop(['Unnamed: 4','Metrics'], axis=1)
            df1 = df.set_index(['Store', 'Store_Location', 'Item','Date']).fillna(0)
            year_list=list(df1.columns)
            df2 = pd.melt(df1, value_vars=year_list , var_name='date',value_name='sold_quantity', ignore_index=False,).reset_index()
            df3 = df2[[ 'date', 'Store','Item','sold_quantity']]
            df3.columns = [ 'date', 'store','item','sales']
            df3['date'] = pd.to_datetime(df3['date'], format = '%d/%m/%Y')
            data= df3#.fillna(0)
            s = pd.to_numeric(data['sales'], errors='coerce')
            data['sales'] = s
            
        elif filename=='NewAV':
            df = pd.read_csv(filename+".csv")
            df = df.drop(['Unnamed: 4','Metrics'], axis=1)
            df1 = df.set_index(['Store', 'Store_Location ', 'Item','Date'])
            year_list=list(df1.columns)
            df2 = pd.melt(df1, value_vars=year_list , var_name='date',value_name='av_quantity', ignore_index=False).reset_index()
            df3 = df2[[ 'date', 'Store','Item','av_quantity']]
            df3.columns = [ 'date', 'store','item','av']
            df3['date'] = pd.to_datetime(df3['date'], format = '%d/%m/%Y')
            data= df3.fillna(0)
            s = pd.to_numeric(data['av'], errors='coerce')
            data['av'] = s
            
        elif filename=='NewWaste':
            df = pd.read_csv(filename+".csv")
            df = df.drop(['Unnamed: 4','Metrics'], axis=1)
            df1 = df.set_index(['Store', 'Store_Location', 'Item','Date']).fillna(0)
            year_list=list(df1.columns)
            df2 = pd.melt(df1, value_vars=year_list , var_name='date',value_name='waste_quantity', ignore_index=False).reset_index()
            df3 = df2[[ 'date', 'Store','Item','waste_quantity']]
            df3.columns = [ 'date', 'store','item','waste']
            df3['date'] = pd.to_datetime(df3['date'], format = '%d/%m/%Y')
            data= df3.fillna(0)
            s = pd.to_numeric(data['waste'], errors='coerce')
            data['waste'] = s
            
        elif filename=='NewStock':
            df = pd.read_csv(filename+".csv", encoding='iso-8859-1')
            df = df.drop(['Unnamed: 4','Metrics'], axis=1)
            df1 = df.set_index(['Store', 'Store_Location', 'Item','Date']).fillna(0)
            year_list=list(df1.columns)
            df2 = pd.melt(df1, value_vars=year_list , var_name='date',value_name='stock_quantity', ignore_index=False).reset_index()
            df3 = df2[[ 'date', 'Store','Item','stock_quantity']]
            df3.columns = [ 'date', 'store','item','stock']
            df3['date'] = pd.to_datetime(df3['date'], format = '%d/%m/%Y')
            data = df3.fillna(0)
            s = pd.to_numeric(data['stock'], errors='coerce')
            data['stock'] = s
            
        if len(req_df)==0:
            req_df=data
            start=str(data['date'].iloc[0])
            end=str(data['date'].iloc[-1])
    
            ideal_dates_df=pd.DataFrame(data=pd.date_range(start=start,end=end),columns=['date'])
            req_df=ideal_dates_df.merge(req_df, on=['date'],how='left')
            req_df=req_df.fillna(0)
                                      
        else:
            req_df = req_df.merge(data, on = ['date','store','item'], how = 'left')
        
    req_df.holiday = req_df.holiday.fillna('normal')
    req_df = req_df.fillna(0)
    req_df['date'] = req_df['date'].astype('datetime64[ns]')
    req_df = req_df.sort_values('date', ascending=True)
    
    req_df['month'] = pd.to_datetime(req_df['date'], format = '%d/%m/%Y').dt.month
    req_df['year'] = pd.to_datetime(req_df['date'], format = '%d/%m/%Y').dt.year
    req_df['week'] = pd.to_datetime(req_df['date'], format = '%d/%m/%Y').dt.week
    req_df['day'] = pd.to_datetime(req_df['date'], format = '%d/%m/%Y').dt.day
    req_df['weekday'] = pd.to_datetime(req_df['date'], format = '%d/%m/%Y').dt.weekday
    req_df['quarter'] = pd.to_datetime(req_df['date'], format = '%d/%m/%Y').dt.quarter
    
    ## Some Arithematic Operations ####
    req_df['correct_sales'] = req_df['sales']/req_df['av']
    req_df['correct_sales'] = req_df['correct_sales'].replace(np.inf,0)
    req_df['correct_sales'] = req_df['correct_sales'].replace(-np.inf,0)
    
    ### ## Aggregating for total level ## ########
    total = req_df.groupby(['date','item']).agg({'sales':'sum','stock':'sum','value':'sum','av':'mean', 'correct_sales':'sum','waste':'sum','month':'mean','year':'mean','day':'mean','weekday':'mean','week':'mean','quarter':'mean'})#.reset_index()#.plot()
    total.reset_index(level=[1],inplace=True)
    
    for j in SKU_codes:
        tat=total[total.item==int(j)]
        tat['sales'] = tat['sales'].replace(0, float('nan')).fillna(tat['sales'].mean())
        tat['value'] = tat['value'].replace(0, float('nan')).fillna(tat['value'].mean())
        tat['price'] = tat['value']/tat['sales']

        tat['potential_sales'] = tat['correct_sales']*tat['av']
        tat['potential_av'] =tat['sales']/ tat['correct_sales']

        tat['potential_av'][tat['potential_av'] > 1.5] = 1.5
        tat['possible_sales'] = tat['potential_av']*tat['sales']
        
        tat.to_csv(save_path+f'Day level sales for {j}.csv')
            
    return None