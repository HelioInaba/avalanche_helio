import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from datetime import datetime
from datetime import timedelta

def main():

    def fetch_data_from_jellyfish(end_point, no_Days):

        if end_point == 'performance':
            url = f'https://jellyfish.steakhut.finance/api/enigma/pool/performance?enigmaAddress=0x144977c1ed62b545a0ceeb0475d2014b78dda8ed&chainId=43114&noDays={no_Days}'
            response = requests.get(url=url)
            response_json = response.json()

            return response_json
        
        if end_point == 'harvests':

            time_stamp_begin = int((datetime.now() - timedelta(days=int(no_Days))).timestamp()*1e3)
            url =f'https://jellyfish.steakhut.finance/api/enigma/pool/harvests?&address=0x144977c1ed62b545a0ceeb0475d2014b78dda8ed&timestamp={time_stamp_begin}'
            response = requests.get(url=url)
            response_json = response.json()

            return response_json

    def transform_to_dataframe(data):
        """
        Transforms a list of nested dictionaries into a pandas DataFrame.
        The DataFrame will have the 'date' as the index and the specified columns.

        Parameters:
            data (list): List of dictionaries containing the data to transform.

        Returns:
            pd.DataFrame: A DataFrame with the transformed data.
        """
        # Flatten the nested dictionaries into a single dictionary for each row
        flat_data = []
        for entry in data:
            flat_entry = {
                'date': entry['date'],
                'tvl': entry['tvl'],
                'token0_price': entry['token0']['price'],
                'token0_tvl': entry['token0']['tvl'],
                'token0_decimals': entry['token0']['decimals'],
                'token0_performance': entry['token0']['performance'],
                'token1_price': entry['token1']['price'],
                'token1_tvl': entry['token1']['tvl'],
                'token1_decimals': entry['token1']['decimals'],
                'token1_performance': entry['token1']['performance'],
                'performance_strategyReturn': entry['performance']['strategyReturn'],
                'performance_holdToken0Return': entry['performance']['holdToken0Return'],
                'performance_holdToken1Return': entry['performance']['holdToken1Return'],
                'performance_hold5050Return': entry['performance']['hold5050Return'],
                'performance_vsConstantRatio': entry['performance']['vsConstantRatio'],
                'performance_vs5050': entry['performance']['vs5050'],
                'performance_vsHoldToken0': entry['performance']['vsHoldToken0'],
                'performance_vsHoldToken1': entry['performance']['vsHoldToken1'],
            }
            flat_data.append(flat_entry)
        
        # Create a DataFrame
        df = pd.DataFrame(flat_data)
        
        # Set the index to 'date'
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Calculate time interval between results
        df['time_delta'] = df.index.to_series().diff()
        df['time_delta'] = [x.days for x in df['time_delta']]

        # Return values in percentage
        for column_name in df.columns:

            if 'performance' in column_name:
                df[column_name] = df[column_name]/100

        df.columns = [x.replace('performance_', '') if ('performance_' in x) else x for x in df.columns]
        
        return df


    def get_performance_data(no_Days):

        fetched_data = fetch_data_from_jellyfish('performance', no_Days)
        df_returns = transform_to_dataframe(fetched_data[0]['dailyData'])

        return df_returns
    

    def get_harvests_data(no_Days):

        fetched_data = fetch_data_from_jellyfish('harvests', no_Days)
        df = pd.DataFrame(fetched_data)
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')
        df = df.set_index('date')
        
        df['apr'] = df['apr']/100
        df['daily_apr'] = df.apr/365
        df['accum_apr'] = df['daily_apr'].cumsum()

        return df

    def plot_histogram(df):

        
        data_diff = df.strategyReturn.diff()
        data_diff = data_diff.dropna()

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        bins = np.histogram(
            data_diff.dropna(), 
            bins=20)[1]

        bins = [float(x) for x in bins]

        fig, ax = plt.subplots(figsize=(16,9))
        sns.histplot(
            data_diff,stat='density',
            bins=20,
            ax=ax)

        sns.lineplot(
            ax=ax,x=bins,
            y=gaussian(data=data_diff,
                        bins=bins),
            color='red')

        textstr = '\n'.join((
            r'$\mu=%.2f$' % (data_diff.mean(), ),
            r'$\mathrm{median}=%.2f$' % (data_diff.median(), ),
            r'$\sigma=%.2f$' % (data_diff.std(), )
            ))

        ax.text(0.1, 0.95, textstr, 
                    horizontalalignment='center', 
                    verticalalignment='top', 
                    transform=ax.transAxes, 
                    bbox=props)

        st.pyplot(plt.gcf())

    def gaussian(data, bins):
        
        return 1/(data.std()*np.sqrt(2*np.pi))*(np.exp(-(bins - data.mean())**2/(2*data.std()**2)))


    def sharpe_ratio(df, n=255):

        daily_returns = df.strategyReturn.diff()/df.time_delta
        daily_std = daily_returns.std()
        daily_avg_return = daily_returns.mean()
        annual_std = daily_std*np.sqrt(n)
        annual_avg_return = daily_avg_return*n

        return annual_avg_return/annual_std

    def sortino_ratio(df, n=255):

        daily_returns = df.strategyReturn.diff()/df.time_delta
        daily_std = daily_returns[daily_returns < 0].std()
        daily_avg_return = daily_returns.mean()
        annual_std = daily_std*np.sqrt(n)
        annual_avg_return = daily_avg_return*n

        return annual_avg_return/annual_std


    def max_drawdown(df):

        accum_return = 1 + df.strategyReturn
        peak = accum_return.expanding(min_periods=1).max()
        drawdown = (accum_return/peak) -1

        return drawdown.min()
    

    def historical_var(df, confidence_level = .95):
        
        daily_returns = df.strategyReturn.diff()/df.time_delta
        daily_returns = daily_returns.dropna()
        
        var = np.percentile(daily_returns,(1-confidence_level))
        return var

    def metrics(df):

        df_metrics = pd.DataFrame(
        index=['Sharpe Ratio','Sortino Ratio', 'Max Drawdown', 'Historical_VaR (95%)'],
        data=[
            sharpe_ratio(df),
            sortino_ratio(df),
            max_drawdown(df),
            historical_var(df)])
        
        df_metrics.columns = ['Value']

        return df_metrics.T
    

    st.title('AVAX-USDC: Autopool Returns')
    no_days = st.text_input(
        label = 'Provide timeframe to be analyzed (in days):',
        value='180')

    df_returns = get_performance_data(no_days)
    df_harvests = get_harvests_data(no_days)

    performance_columns = [
        'strategyReturn', 'holdToken0Return',
        'holdToken1Return', 'hold5050Return',
        'vsConstantRatio', 'vs5050',
        'vsHoldToken0', 'vsHoldToken1']
    
    df_returns['harvest_return'] = df_harvests.daily_apr.cumsum().loc[df_returns.index]
    
    st.line_chart(df_returns[performance_columns])
    st.subheader('Strategy Return: Risk/Return Metrics')
    st.dataframe(data = metrics(df_returns))
    st.subheader('Distribution of Daily Strategy Return')
    plot_histogram(df_returns)

    st.subheader('Historical and Daily Accumulated APR')
    st.line_chart(df_harvests[['apr','accum_apr']])

    st.subheader('Strategy Return vs Daily Accumulated APR')
    st.line_chart(df_returns[['harvest_return','strategyReturn']])

if __name__ == '__main__':
    main()