import datetime

from dateutil.relativedelta import relativedelta

import db_calls as db
import pandas as pd
import numpy as np
import geopandas as gpd


def intersect_atributes_null(data):
    print('Number of atributes nulls before:')
    print(data.isnull().values.sum())
    print('processing...')
    colums = data.columns[data.isna().any()].tolist()
    for colum in colums:
        for i, empty in enumerate(data[colum].isna()):
            if empty:
                avg_neight = db.get_avg_atribute_from_neighbor(data['geometry'][i], colum)
                if avg_neight is not None:
                    data.at[i, colum] = avg_neight
    print('Number of atributes nulls after:')
    print(data.isnull().values.sum())
    return data


def complete_dates_meteo():
    name_stations = db.get_table_without_geometry('select m."Name" as name from meteo m group by m."Name"', 'GeoHealth')
    dataframe = db.get_table_meteo()
    data_to_add = {column: [] for column in dataframe.columns.values}
    max_date = max(dataframe['Date'])
    min_date = min(dataframe['Date'])
    print(len(dataframe))
    for name in name_stations['name']:
        print(name)
        colums = db.get_table_meteo_by_name(name)
        actual_date = min_date
        while actual_date < max_date:
            if colums[colums.Date == actual_date].empty:
                print(actual_date)
                data_to_add['geometry'].append(colums['geometry'][0])
                data_to_add['Name'].append(colums['Name'][0])
                data_to_add['Province'].append(colums['Province'][0])
                data_to_add['Altitude'].append(colums['Altitude'][0])
                data_to_add['Date'].append(actual_date)
            actual_date += datetime.timedelta(days=1)
    size = len(data_to_add['Name'])
    for col in dataframe.columns.values:
        data_to_add[col] = [None] * size if not data_to_add[col] else data_to_add[col]
    data_to_add = pd.DataFrame(data_to_add)
    dataframe = pd.concat([dataframe, data_to_add], sort=True)
    print(len(dataframe))
    db.update_table_to_analysis(dataframe, 'meteo_full')


def complete_dates_pollution():
    name_stations = db.get_table_without_geometry('select m."Name" as name from pollution m group by m."Name"',
                                                  'GeoHealth')
    dataframe = db.get_table_pollution()
    data_to_add = {column: [] for column in dataframe.columns.values}
    max_date = max(dataframe['Date'])
    min_date = min(dataframe['Date'])
    print(len(dataframe))
    for name in name_stations['name']:
        print(name)
        colums = db.get_table_pollution_by_name(name)
        actual_date = min_date
        while actual_date < max_date:
            if colums[colums.Date == actual_date].empty:
                print(actual_date)
                data_to_add['geometry'].append(colums['geometry'][0])
                data_to_add['Name'].append(colums['Name'][0])
                data_to_add['Province'].append(colums['Province'][0])
                data_to_add['Municipality'].append(colums['Municipality'][0])
                data_to_add['Date'].append(actual_date)
            actual_date += datetime.timedelta(days=1)
    size = len(data_to_add['Name'])
    for col in dataframe.columns.values:
        data_to_add[col] = [None] * size if not data_to_add[col] else data_to_add[col]
    data_to_add = pd.DataFrame(data_to_add)
    dataframe = pd.concat([dataframe, data_to_add], sort=True)
    print(len(dataframe))
    db.update_table_to_analysis(dataframe, 'pollution_full')


def interpolate_pollution():
    colums = ['NO2', 'O3', 'PART']
    name_stations = db.get_table_without_geometry('select m."Name" as name from pollution m group by m."Name"',
                                                  'GeoHealth')
    dataframe = pd.DataFrame()
    for name in name_stations['name']:
        print(name)
        data_name = db.get_table_pollution_by_name(name)
        print(f'Nulos antes: {data_name[colums].isnull().sum().sum()}')
        data_name = data_name.sort_values(by='Date')
        data_name[colums[0]] = data_name[colums[0]].interpolate(method='linear', limit_direction='both', limit=5,
                                                                axis=0)
        data_name[colums[1]] = data_name[colums[1]].interpolate(method='linear', limit_direction='both', limit=5,
                                                                axis=0)
        data_name[colums[2]] = data_name[colums[2]].interpolate(method='linear', limit_direction='both', limit=5,
                                                                axis=0)
        print(f'Nulos despues: {data_name[colums].isnull().sum().sum()}')
        dataframe = pd.concat([dataframe, data_name], sort=True)
    db.update_table_to_geohealth(dataframe, 'pollution')


def interpolate_meteo():
    colums = ['Average_Wind_Speed', 'Precipitation', 'Average_Temperature']
    name_stations = db.get_table_without_geometry('select m."Name" as name from meteo m group by m."Name"', 'GeoHealth')
    dataframe = pd.DataFrame()
    for name in name_stations['name']:
        print(name)
        data_name = db.get_table_meteo_by_name(name)
        print(f'Nulos antes: {data_name[colums].isnull().sum().sum()}')
        data_name = data_name.sort_values(by='Date')
        data_name[colums[0]] = data_name[colums[0]].interpolate(method='linear', limit_direction='both', limit=3,
                                                                axis=0)
        data_name[colums[1]] = data_name[colums[1]].interpolate(method='linear', limit_direction='both', limit=3,
                                                                axis=0)
        data_name[colums[2]] = data_name[colums[2]].interpolate(method='linear',
                                                                limit_direction='both',
                                                                axis=0)
        print(f'Nulos despues: {data_name[colums].isnull().sum().sum()}')
        dataframe = pd.concat([dataframe, data_name], sort=True)
    db.update_table_to_geohealth(dataframe, 'meteo')


def delete_outliers_values(table, atributes):
    umbral = 10.0
    columns = table.columns.values
    for atr in atributes:
        table[f'z_scores_{atr}'] = (table[atr] - table[atr].mean()) / table[atr].std(ddof=0)
    for i in range(0, len(table)):
        for atr in atributes:
            if table[f'z_scores_{atr}'][i] > umbral:
                table.at[i, atr] = None
    return table[columns]


def delete_outliers_values_pollution():
    names = db.get_station_pollution_names()
    data = pd.DataFrame()
    for name in names['name']:
        print(name)
        col = db.get_table_pollution_by_name(name)
        col = delete_outliers_values(col, ['NO2', 'O3', 'PART'])
        data = pd.concat([data, col], sort=True)
    db.update_table_to_geohealth(data, 'pollution')


def normalize_dataframe(data, colums_to_normalize):
    copy = data.copy()
    for colum in colums_to_normalize:
        mx = copy[colum].max()
        mn = copy[colum].min()
        copy[colum] = copy[colum].map(lambda x: (x - mn) / (mx - mn))
    return copy


def get_pollution_in_month():
    df_p = db.get_table_pollution()
    res = pd.DataFrame(columns=df_p.columns.to_list())
    date_from = df_p['Date'].min().date()
    date_to = date_from + relativedelta(months=1)
    date_max = df_p['Date'].max().date()
    while date_from < date_max:
        df = db.get_table_pollution_from_dates(date_from, date_to)
        df['Date'] = pd.to_datetime(date_from)
        res = res.append(df)
        date_from = date_from + relativedelta(months=1)
        date_to = date_to + relativedelta(months=1)
    return res


def intersect_atributes_null_pollution(data, colums):
    print('Number of atributes nulls before:')
    print(data[colums].isnull().values.sum())
    print('processing...')
    for colum in colums:
        for i, empty in enumerate(data[colum].isna()):
            if empty:
                avg = db.get_pollution_intersect(data['Date'][i], data['Name'][i], colum)
                if avg is not None:
                    data.at[i, colum] = avg
    print('Number of atributes nulls after:')
    print(data[colums].isnull().values.sum())
    return data


def expand_pollution_in_grid(grid, pollution):
    gases = ['SH2', 'SO2', 'NO2', 'CO', 'PART', 'O3']
    colums = gases[:]
    colums.append('geometry')
    res = []
    geom = []
    lenght = len(grid)
    for number, point in enumerate(grid['geometry']):
        print(f'Precent: {(number / lenght) * 100}', end='\r')
        geom.append(point)
        distances = []
        values = []
        for i, station in enumerate(pollution['geometry']):
            distances.append(point.distance(station))
            values.append(np.array(pollution[gases].iloc[[i]].values[0]))
        distances = np.array(distances)
        values = np.array(values)
        distances = distances / distances.sum()
        distances = distances.reshape(len(distances), 1)
        values = np.nansum(values / distances, axis=0)
        res.append(values)
    res = pd.DataFrame(res, columns=gases)
    res['geometry'] = geom
    return res
