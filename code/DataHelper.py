from fredapi import Fred
import pandas as pd
import datetime

API_KEY = '5f1fcf42828a0577bb4c032324131951'
FREQUENCY = 'q'
loader = Fred(API_KEY)

column_config = {
    'PCECC96':'pch',
    'TWEXM': 'pch', #dollar
    'HOUST': 'pch', #housing starts
    'TOTALSL': 'pch', #total consumer credit
    'CIVPART': 'pch', #civilian labor force participation rate
    'POP': 'pch', #Total Population: All Ages including Armed Forces Overseas
    'UNRATE': 'pch', #unemployment rate
    'PSAVERT': 'pch', #savings rate
    'W875RX1': 'pch', #real personal income
    'TOTALSA': 'pch', #total vehicle sales
    'M2' : 'pch', #m2 money supply
    #'BAA10Y' : 'lin', #corporate bond spread
    'FF' : 'pch', #fed funds rate
    'DGS10' : 'pch', #10Y yield
    'UMCSENT' : 'pch', #consumer sentiment
    'USTRADE' : 'pch', #All Employees: Retail Trade
    'CPIAUCSL' : 'pch', #Consumer Price Index for All Urban Consumers: All Items
    'WTISPLC' : 'pch', #spot price wti
    'WILL5000INDFC': 'pch'
}

target = ['PCECC96', 'pch']

info = {'frequency': 'Quarterly',
        'units': 'lin'}

def get_experts(load_from_file):

    if load_from_file:
        experts = pd.read_csv('../data/ExpertsProcessed.csv', index_col='Date')
        experts.index = pd.to_datetime(experts.index)
    else:
        experts = pd.read_csv('../data/ExpertsMean_In.csv')
        experts = experts.set_index(((experts["QUARTER"] - 1) * 3 + 1).map(str) + '/1/' + experts["YEAR"].map(str))
        experts.index = pd.to_datetime(experts.index)
        experts = experts.rename_axis('Date')

        experts['fwd1'] = 100 * (experts['RCONSUM3'] - experts['RCONSUM2']) / experts['RCONSUM2']
        experts['fwd2'] = 100 * (experts['RCONSUM4'] - experts['RCONSUM3']) / experts['RCONSUM3']
        experts['fwd3'] = 100 * (experts['RCONSUM5'] - experts['RCONSUM4']) / experts['RCONSUM4']
        experts['fwd4'] = 100 * (experts['RCONSUM6'] - experts['RCONSUM5']) / experts['RCONSUM5']

        experts.to_csv('../data/ExpertsProcessed.csv')

    return experts


def get_all_data(load_from_file):

    if load_from_file:
        df = pd.read_csv('../data/input_data.csv', index_col='Date')
        df.index = pd.to_datetime(df.index)
    else:
        df = pd.DataFrame()
        i = 0
        for key in column_config.keys():
            one_series = loader.get_series(key, units=column_config[key], frequency=FREQUENCY)
            df.insert(i,key, one_series)
            print('loaded: ' + key + ' start date: ' + str(one_series.index[0]))
            i += 1

        df = get_deltas(df)
        df = df.dropna()
        df = df.rename_axis('Date')
        df.to_csv('../data/input_data.csv')

    return(df)

def get_deltas(df):

    df_shifted = df.shift(1)
    df_deltas = df - df_shifted
    num_cols = df.shape[1]

    for i in range(num_cols):
        df.insert(i + num_cols,df.columns[i] + '_delta', df_deltas.iloc[:, i])
        i += 1

    return df

#get_all_data(False)
