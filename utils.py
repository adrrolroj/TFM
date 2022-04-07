import datetime

import torch
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler

from pykrige.ok import OrdinaryKriging
import db_calls as db
from scipy.spatial import cKDTree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from neural_model.net_model import netModel, netModel_I


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
    db.update_table(dataframe, 'meteo_full', 'GeoHealth')


def complete_dates_pollen():
    name_stations = db.get_table_without_geometry('select m."id" as id from pollen m group by m."id"', 'GeoHealth')
    dataframe = db.get_table_pollen()
    data_to_add = {column: [] for column in dataframe.columns.values}
    print(len(dataframe))
    for id in name_stations['id']:
        print(id)
        colums = db.get_table_pollen_by_id(id)
        max_date = max(colums['date'])
        min_date = min(colums['date'])
        actual_date = min_date
        while actual_date < max_date:
            if colums[colums.date == actual_date].empty:
                print(actual_date)
                for column in dataframe.columns.values:
                    if column == 'id':
                        data_to_add['id'].append(id)
                    elif column == 'geometry':
                        data_to_add['id'].append(colums['geometry'][0])
                    elif column == 'ccaa':
                        data_to_add['id'].append(colums['ccaa'][0])
                    elif column == 'province':
                        data_to_add['id'].append(colums['province'][0])
                    elif column == 'name':
                        data_to_add['id'].append(colums['name'][0])
                    elif column == 'date':
                        data_to_add['id'].append(actual_date)
                    else:
                        data_to_add[column].append(None)
            actual_date += datetime.timedelta(days=1)
    data_to_add = pd.DataFrame(data_to_add)
    dataframe = pd.concat([dataframe, data_to_add], sort=True)
    print(len(dataframe))
    db.update_table(dataframe, 'pollen_full', 'GeoHealth')


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
    db.update_table(dataframe, 'pollution_full', 'GeoHealth')


def interpolate_pollution_from_dates():
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


def interpolate_pollen_from_dates():
    columns = ['Alnus', 'Alternaria', 'Artemisa',
               'Betula', 'Carex', 'Castanea', 'Cupresaceas', 'Fraxinus', 'Gramineas',
               'Mercurialis', 'Morus', 'Olea', 'Palmaceas', 'Pinus', 'Plantago', 'Platanus',
               'Populus', 'Amarantaceas', 'Quercus', 'Rumex', 'Ulmus', 'Urticaceas']
    ids = db.get_table_without_geometry('select m."id" as id from pollen m group by m."id"',
                                        'GeoHealth')
    dataframe = pd.DataFrame()
    for id in ids['id']:
        print(id)
        data_name = db.get_table_pollen_by_id(id)
        print(f'Nulos antes: {data_name[columns].isnull().sum().sum()}')
        data_name = data_name.sort_values(by='date')
        for column in columns:
            data_name[column] = data_name[column].interpolate(method='linear', limit_direction='both', limit=7,
                                                              axis=0)
        print(f'Nulos despues: {data_name[columns].isnull().sum().sum()}')
        dataframe = pd.concat([dataframe, data_name], sort=True)
    db.update_table_to_geohealth(dataframe, 'pollen')


def interpolate_meteo_from_dates():
    colums = ['Average_Wind_Speed', 'Precipitation', 'Average_Temperature']
    name_stations = db.get_table_without_geometry('select m."Name" as name from meteo m group by m."Name"', 'GeoHealth')
    dataframe = pd.DataFrame()
    for name in name_stations['name']:
        print(name)
        data_name = db.get_table_meteo_by_name(name)
        print(f'Nulos antes: {data_name[colums].isnull().sum().sum()}')
        data_name = data_name.sort_values(by='Date')
        data_name[colums[0]] = data_name[colums[0]].interpolate(method='linear', limit_direction='both',
                                                                axis=0)
        data_name[colums[1]] = data_name[colums[1]].interpolate(method='linear', limit_direction='both',
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


def interpolate_pollen_censal_by_nearest_points(date):
    data = db.get_table_censal_with_near_pollen_stations_in_date(date)
    columns = ['Alnus', 'Alternaria', 'Artemisa',
               'Betula', 'Carex', 'Castanea', 'Cupresaceas', 'Fraxinus', 'Gramineas',
               'Mercurialis', 'Morus', 'Olea', 'Palmaceas', 'Pinus', 'Plantago', 'Platanus',
               'Populus', 'Amarantaceas', 'Quercus', 'Rumex', 'Ulmus', 'Urticaceas']
    data_group = data.groupby(by=["cusec"])
    result = {column: [] for column in columns}
    result['geometry'] = []
    result['cusec'] = []
    for group_cusec in data_group.groups.keys():
        group = data_group.get_group(group_cusec)
        group.reset_index(drop=True, inplace=True)
        distances = group['distance']
        for pollen in columns:
            level_pollen = group[pollen]
            value = float(sum([float(level_pollen[i]) / float(distances[i]) for i in range(0, 3)])) / float(
                sum([1.0 / float(distances[i]) for i in range(0, 3)]))
            result[pollen].append(value)
        result['cusec'].append(group_cusec)
        result['geometry'].append(group['geometry'][0])
    result = pd.DataFrame.from_dict(result)
    db.update_table_to_geohealth(result, 'temp_table')
    result = db.get_table('select * from temp_table', 'GeoHealth')
    return result


def voronoi_pollen(date):
    data = db.get_voronoi_censal_pollen(date)
    data = data.fillna(0.0)
    return data


def interpolate_pollution_censal_by_nearest_points(gas, date):
    data = db.get_censal_pollution_nearest_points(gas, date)
    data_group = data.groupby(by=["cusec"])
    result = {'geometry': [], 'cusec': [], 'values': []}
    for group_cusec in data_group.groups.keys():
        group = data_group.get_group(group_cusec)
        group.reset_index(drop=True, inplace=True)
        ditances = group['distance']
        level_gas = group[gas]
        value = float(sum([float(level_gas[i]) / float(ditances[i]) for i in range(0, 3)])) / float(
            sum([1.0 / float(ditances[i]) for i in range(0, 3)]))
        result['geometry'].append(group['geometry'][0])
        result['values'].append(value)
        result['cusec'].append(group_cusec)
    result = pd.DataFrame.from_dict(result)
    return result


def delete_outliers_values_pollution():
    names = db.get_station_pollution_names()
    data = pd.DataFrame()
    for name in names['name']:
        print(name)
        col = db.get_table_pollution_by_name(name)
        col = delete_outliers_values(col, ['NO2', 'O3', 'PART'])
        data = pd.concat([data, col], sort=True)
    db.update_table_to_geohealth(data, 'pollution')


def get_pollution_in_weeks():
    minmax_date = db.get_table_without_geometry('select min("Date"), max("Date") from pollution', 'GeoHealth')
    min_date = minmax_date['min'][0]
    max_date = minmax_date['max'][0]
    result = pd.DataFrame()
    while min_date < max_date:
        to_date = min_date + datetime.timedelta(weeks=1)
        print(str(to_date))
        data = db.get_table_pollution_from_dates(min_date, to_date)
        min_date += datetime.timedelta(weeks=1)
        result = pd.concat([result, data])
    db.update_table(result, 'pollution_weeks', 'GeoHealth')
    return result


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


def get_scaler(max_element, min_element):
    scaler = MinMaxScaler()
    scaler.fit([[max_element], [min_element]])
    return scaler


def kriging_interpolation(data_input, coordinates_to_out):  # ambos dataframe
    OK = kringing(data_input)
    result = []
    for index, point in coordinates_to_out.iterrows():
        z, ss = OK.execute("points", np.array(point.x), np.array(point.y))
        result.append(z.item(0))
    return result


def kriging_interpolation_method_pollen(date):
    columns = ['Alnus', 'Alternaria', 'Artemisa',
               'Betula', 'Carex', 'Castanea', 'Cupresaceas', 'Fraxinus', 'Gramineas',
               'Mercurialis', 'Morus', 'Olea', 'Palmaceas', 'Pinus', 'Plantago', 'Platanus',
               'Populus', 'Amarantaceas', 'Quercus', 'Rumex', 'Ulmus', 'Urticaceas']
    data = db.get_x_y_value_from_pollen(date)
    points = db.get_x_y_cusec_from_censal()
    for column in columns:
        data_input = pd.DataFrame({'x': data.x.values, 'y': data.y.values, 'value': data[column].values})
        print(data_input.head())
        values = kriging_interpolation(data_input, points)
        points[column] = np.array(values)
    db.update_table_to_geohealth(points, 'temp_table')
    points = db.get_table('select * from temp_table', 'GeoHealth')
    return points


def kriging_interpolation_method(gas, date):
    data = db.get_x_y_value_from_pollution(gas, date)
    points = db.get_x_y_cusec_from_censal()
    values = kriging_interpolation(data, points)
    points['values'] = np.array(values)
    db.update_table_to_geohealth(points, 'temp_table')
    points = db.get_table('select * from temp_table', 'GeoHealth')
    return points


def lineal_interpolation_method(gas, date):
    data = interpolate_pollution_censal_by_nearest_points(gas, date)
    db.update_table_to_geohealth(data, 'temp_table')
    data = db.get_table('select * from temp_table', 'GeoHealth')
    return data


def parameters_net_method(gas, date):
    date_time_obj = datetime.datetime.strptime(date, '%m-%d-%Y')
    INPUT_SIZE = 10

    model = netModel(input_size=INPUT_SIZE)
    model.load_state_dict(torch.load(f'neural_model/models/{gas}_model.pth'))
    model.eval()

    points = db.get_center_density_censal(str(date_time_obj))
    cusec_points = []
    geometry = []
    values = []
    points = normalize_dataframe_train(points)
    for index, point in points.iterrows():
        input_point = transform_dataframe_to_tensor(point)
        with torch.no_grad():
            value = model(input_point).detach().numpy()[0][0]
        cusec_points.append(point['cusec'])
        geometry.append(point['geometry'])
        values.append(value)

    cusec_points = np.array(cusec_points)
    df_grid = pd.DataFrame({'cusec': cusec_points, 'geometry': geometry, 'values': values})
    db.update_table_to_geohealth(df_grid, 'temp_table')
    df_grid = db.get_table('select * from temp_table', 'GeoHealth')
    return df_grid


def interpolation_net_method(gas, date):
    data = interpolate_pollution_censal_by_net(gas, date)
    db.update_table_to_geohealth(data, 'temp_table')
    data = db.get_table('select * from temp_table', 'GeoHealth')
    return data


def methods_combinated(gas, date):
    lineal = lineal_interpolation_method(gas, date)
    lineal.sort_values(by=['cusec'], ascending=False, inplace=True)
    kriging = kriging_interpolation_method(gas, date)
    kriging.sort_values(by=['cusec'], ascending=False, inplace=True)
    param_net = parameters_net_method(gas, date)
    param_net.sort_values(by=['cusec'], ascending=False, inplace=True)
    inter_net = interpolation_net_method(gas, date)
    inter_net.sort_values(by=['cusec'], ascending=False, inplace=True)
    result = {'geometry': [], 'cusec': [], 'values': []}
    values = (lineal['values'].values + kriging['values'].values + param_net['values'].values + inter_net[
        'values'].values) / 4.0
    [result['values'].append(val) for val in values]
    [result['cusec'].append(val) for val in lineal['cusec'].values]
    [result['geometry'].append(val) for val in lineal['geometry'].values]
    result = pd.DataFrame.from_dict(result)
    db.update_table_to_geohealth(result, 'temp_table')
    result = db.get_table('select * from temp_table', 'GeoHealth')
    return result


def interpolate_pollution_censal_by_net(gas, date):
    data = db.get_censal_for_net_interpolation(gas, date)
    INPUT_SIZE = 10

    model_I = netModel_I(input_size=INPUT_SIZE)
    model_I.load_state_dict(torch.load(f'neural_model/models/{gas}_model_inter.pth'))
    model_I.eval()

    data_group = data.groupby(by=["cusec"])
    result = {'geometry': [], 'cusec': [], 'values': []}
    for group_cusec in data_group.groups.keys():
        group = data_group.get_group(group_cusec)
        group.reset_index(drop=True, inplace=True)
        ditances = group['distance']
        level_gas = group[gas]

        dataframe = pd.DataFrame({f'1_{gas}': [level_gas[0]], f'2_{gas}': [level_gas[1]], f'3_{gas}': [level_gas[2]],
                                  f'1_dist_{gas}': [ditances[0]], f'2_dist_{gas}': [ditances[1]],
                                  f'3_dist_{gas}': [ditances[2]], 'neigh_density': [group['neigh_density'][0]],
                                  'wind': [group['wind'][0]], 'rain': [group['rain'][0]],
                                  'altitude': [group['altitude'][0]], })
        input_point = normalize_dataframe_train_interpol(dataframe, gas)
        with torch.no_grad():
            value_model_I = model_I(input_point).detach().numpy().item(0)
        result['geometry'].append(group['geometry'][0])
        result['values'].append(value_model_I)
        result['cusec'].append(group_cusec)
    result = pd.DataFrame.from_dict(result)
    return result


def kringing(data_input):
    OK = OrdinaryKriging(
        np.array(data_input.x),
        np.array(data_input.y),
        np.array(data_input.value),
        variogram_model="spherical",
        verbose=False,
        enable_plotting=False,
        coordinates_type='geographic'
    )
    return OK


def update_pollution_nearest3stations_by_gas(gas):
    df = db.get_table(f'select * from pollution p where p."{gas}" is not null', 'GeoHealth')
    cols_gas = []
    cols_dist = []
    for index, row in df.iterrows():
        date = f"'{row['Date']}'"
        name = f"'{row['Name']}'"
        query = f'select * from pollution p where p."{gas}" is not null and p."Date"={date} and p."Name"!={name}'
        print(index)
        temp = db.get_table(query, 'GeoHealth')
        nA = np.array(list((row.geometry.x, row.geometry.y)))
        nB = np.array(list(temp.geometry.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(nB)
        dist, idx = btree.query(nA, k=3)
        gdB_nearest = temp.iloc[idx].reset_index(drop=True)
        cols_dist.append(dist)
        cols_gas.append(gdB_nearest[gas].values)

    cols_dist = np.array(cols_dist)
    cols_gas = np.array(cols_gas)
    df[f'1_{gas}'] = cols_gas[:, 0]
    df[f'2_{gas}'] = cols_gas[:, 1]
    df[f'3_{gas}'] = cols_gas[:, 2]
    df[f'1_dist_{gas}'] = cols_dist[:, 0]
    df[f'2_dist_{gas}'] = cols_dist[:, 1]
    df[f'3_dist_{gas}'] = cols_dist[:, 2]

    db.update_table_to_geohealth(df, f'pollution_{gas}_train')


def normalize_dataframe_train(dataframe):
    df_max_min = db.get_min_max_censal()
    scaler_density_neigh = get_scaler(df_max_min['max_dn'][0], df_max_min['min_dn'][0])
    scaler_density = get_scaler(df_max_min['max_d'][0], df_max_min['min_d'][0])
    scaler_x = get_scaler(df_max_min['max_x'][0], df_max_min['min_x'][0])
    scaler_y = get_scaler(df_max_min['max_y'][0], df_max_min['min_y'][0])
    scaler_day = get_scaler(366, 1)
    scaler_year = get_scaler(2021, 2010)
    scaler_temp = get_scaler(df_max_min['max_temp'][0], df_max_min['min_temp'][0])
    scaler_wind = get_scaler(df_max_min['max_wind'][0], 0.0)
    scaler_rain = get_scaler(df_max_min['max_rain'][0], 0.0)
    scaler_altitude = get_scaler(df_max_min['max_altitude'][0], 0.0)
    dataframe['altitude'] = dataframe['altitude'].astype('float', copy=False)
    for index, row in dataframe.iterrows():
        dataframe.at[index, 'x'] = scaler_x.transform(np.array(row['x']).reshape(-1, 1))
        dataframe.at[index, 'y'] = scaler_y.transform(np.array(row['y']).reshape(-1, 1))
        dataframe.at[index, 'doy'] = scaler_day.transform(np.array(row['doy']).reshape(-1, 1))
        dataframe.at[index, 'year'] = scaler_year.transform(np.array(row['year']).reshape(-1, 1))
        dataframe.at[index, 'density_neigh'] = scaler_density_neigh.transform(
            np.array(row['density_neigh']).reshape(-1, 1))
        dataframe.at[index, 'density'] = scaler_density.transform(np.array(row['density']).reshape(-1, 1))
        dataframe.at[index, 'temp'] = scaler_temp.transform(np.array(row['temp']).reshape(-1, 1))
        dataframe.at[index, 'wind'] = scaler_wind.transform(np.array(row['wind']).reshape(-1, 1))
        dataframe.at[index, 'altitude'] = scaler_altitude.transform(np.array(row['altitude']).reshape(-1, 1))
        dataframe.at[index, 'rain'] = scaler_rain.transform(np.array(row['rain']).reshape(-1, 1))
    return dataframe


def transform_dataframe_to_tensor(dataframe):
    input_point = list()
    x = torch.tensor(np.array(dataframe['x'], dtype=float).reshape(-1, 1))
    input_point.append(x.float())  # X
    y = torch.tensor(np.array(dataframe['y'], dtype=float).reshape(-1, 1))
    input_point.append(y.float())  # Y
    doy = torch.tensor(np.array(dataframe['doy'], dtype=float).reshape(-1, 1))
    input_point.append(doy.float())  # day
    year = torch.tensor(np.array(dataframe['year'], dtype=float).reshape(-1, 1))
    input_point.append(year.float())  # year
    denst_n = torch.tensor(np.array(dataframe['density_neigh'], dtype=float).reshape(-1, 1))
    input_point.append(denst_n.float())  # density neigh
    denst = torch.tensor(np.array(dataframe['density'], dtype=float).reshape(-1, 1))
    input_point.append(denst.float())  # density
    temp = np.array(dataframe['temp'], dtype=float).reshape(-1, 1)
    temp[np.isnan(temp)] = 0.0
    temp = torch.tensor(temp)
    wind = np.array(dataframe['wind'], dtype=float).reshape(-1, 1)
    wind[np.isnan(wind)] = 0.0
    wind = torch.tensor(wind)
    rain = np.array(dataframe['rain'], dtype=float).reshape(-1, 1)
    rain[np.isnan(rain)] = 0.0
    rain = torch.tensor(rain)
    altitude = np.array(dataframe['altitude'], dtype=float).reshape(-1, 1)
    altitude[np.isnan(altitude)] = 0.0
    altitude = torch.tensor(altitude)
    input_point.append(temp.float())  # temp
    input_point.append(wind.float())  # wind
    input_point.append(rain.float())  # rain
    input_point.append(altitude.float())  # altitude
    return input_point


def normalize_dataframe_train_interpol(dataframe, gas):
    input_point = list()
    dist_1 = torch.tensor(np.array(dataframe[f'1_dist_{gas}'], dtype=float).reshape(-1, 1))
    input_point.append(dist_1.float())
    dist_2 = torch.tensor(np.array(dataframe[f'2_dist_{gas}'], dtype=float).reshape(-1, 1))
    input_point.append(dist_2.float())
    dist_3 = torch.tensor(np.array(dataframe[f'3_dist_{gas}'], dtype=float).reshape(-1, 1))
    input_point.append(dist_3.float())
    max_value = [193.5 if gas == 'NO2' else 337.7 if gas == 'PART' else 224]
    scaler_gas = get_scaler(max_value[0], 0.0)
    gas_1 = torch.tensor(scaler_gas.transform(np.array(dataframe[f'1_{gas}']).reshape(-1, 1)))
    input_point.append(gas_1.float())
    gas_2 = torch.tensor(scaler_gas.transform(np.array(dataframe[f'2_{gas}']).reshape(-1, 1)))
    input_point.append(gas_2.float())
    gas_3 = torch.tensor(scaler_gas.transform(np.array(dataframe[f'3_{gas}']).reshape(-1, 1)))
    input_point.append(gas_3.float())
    scaler_density = get_scaler(45000.0, 0.0)
    neigh_density = torch.tensor(scaler_density.transform(np.array(dataframe[f'neigh_density']).reshape(-1, 1)))
    input_point.append(neigh_density.float())
    scaler_wind = get_scaler(24.0, 0.0)
    wind = scaler_wind.transform(np.array(dataframe[f'wind']).reshape(-1, 1))
    wind[np.isnan(wind)] = 0.0
    input_point.append(torch.tensor(wind).float())
    scaler_rain = get_scaler(296.0, 0.0)
    rain = scaler_rain.transform(np.array(dataframe[f'rain']).reshape(-1, 1))
    rain[np.isnan(rain)] = 0.0
    input_point.append(torch.tensor(rain).float())
    scaler_altitude = get_scaler(1201.0, 0.0)
    altitude = scaler_altitude.transform(np.array(dataframe[f'altitude']).reshape(-1, 1))
    altitude[np.isnan(altitude)] = 0.08
    input_point.append(torch.tensor(altitude).float())
    return input_point


def compare_interpolation_with_model(model, model_I, name_stations, dates, gas):
    stations = pd.DataFrame()
    model.eval()
    model_I.eval()
    for i, date in enumerate(dates):
        data_input_OK = db.get_geo_val_from_pollution_in_not_stations(gas, date, name_stations)
        OK = kringing(data_input_OK)
        real_values = []
        interpolated_values = []
        kringing_values = []
        model_values = []
        model_I_values = []
        combinated_values = []
        for station_name in name_stations:
            # Real
            value_real = db.get_value_pollution_station_date(station_name, date, gas)
            if i == 0:
                stations = pd.concat([stations, db.get_station_pollution_by_name(station_name)])
            # Interpolacion
            data = db.get_pollution_interpolated_in_station(station_name, date, gas)
            ditances = data['distance']
            level_gas = data[gas]
            value_interpolated = float(sum([float(level_gas[i]) / float(ditances[i]) for i in range(0, 3)])) / float(
                sum([1.0 / float(ditances[i]) for i in range(0, 3)]))
            # Modelo
            input_dataframe = db.get_input_from(station_name, date)
            input_dataframe = normalize_dataframe_train(input_dataframe)
            input_test = transform_dataframe_to_tensor(input_dataframe)
            with torch.no_grad():
                value_model = model(input_test).detach().numpy().item(0)
            # Modelo Interpol
            try:
                input_dataframe = db.get_input_from_interpol(station_name, date, gas)
                input_dataframe = normalize_dataframe_train_interpol(input_dataframe, gas)
                with torch.no_grad():
                    value_model_I = model_I(input_dataframe).detach().numpy().item(0)
            except:
                value_model_I = 12.0
            # Kringing
            point = db.get_table_pollution_by_name_date_gas(station_name, date, gas)
            value_kringing, ss = OK.execute("points", np.array(point.geometry.x), np.array(point.geometry.y))
            try:
                value_kringing = value_kringing.item(0)
            except:
                value_kringing = 12.0
            kringing_values.append(value_kringing)
            # combinated
            value_combinated = (value_model + value_interpolated + value_kringing + value_model_I) / 4.0
            combinated_values.append(value_combinated)
            real_values.append(value_real)
            interpolated_values.append(value_interpolated)
            model_values.append(value_model)
            model_I_values.append(value_model_I)
        # GRAFICO COMPARATIVO
        plt.figure(i)
        plt.title(f'Comparativa interpolacion, red neuronal, fecha: {date}')
        plt.xticks(np.array(range(0, len(name_stations))), name_stations)
        plt.plot(range(0, len(real_values)), real_values, '-o', label='Medicion real')
        plt.plot(range(0, len(model_values)), model_values, '-o', label='Red 1')
        plt.plot(range(0, len(interpolated_values)), interpolated_values, '-o', label='Interpolacion')
        plt.plot(range(0, len(kringing_values)), kringing_values, '-o', label='Kriging')
        plt.plot(range(0, len(combinated_values)), combinated_values, '-o', label='Combinacion')
        plt.plot(range(0, len(model_I_values)), model_I_values, '-o', label='Red 2')
        plt.legend(loc='upper right')

    # MAPA CON LAS ESTACIONES
    fig, ax = plt.subplots()
    region = db.get_table_health_districts_region()
    all_stations = db.get_station_pollution()
    region.plot(ax=ax, color='white', edgecolor='black')
    all_stations.plot(ax=ax, marker='o', color='blue', markersize=5)
    stations.plot(ax=ax, marker='o', color='red', markersize=7)
    for x, y, label in zip(stations.geometry.x, stations.geometry.y, stations.Name):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")
    plt.show()


def compare_interpolation_with_model_stadistics(model, model_I, name_stations, gas):
    model.eval()
    model_I.eval()
    dataframe = db.get_table_pollution_in_stations(name_stations, gas)
    interpolated_sum_error = []
    model_sum_error = []
    model_i_sum_error = []
    kringing_sum_error = []
    combinated_sum_error = []
    grouped = dataframe.groupby('Name')
    for name, group in grouped:
        real_values = []
        interpolated_values = []
        model_values = []
        model_i_values = []
        kringing_values = []
        combinated_values = []
        for count, (index, row) in enumerate(group.iterrows()):
            real_values.append(row[gas])
            # kringing
            data_input_OK = db.get_geo_val_from_pollution_in_not_stations(gas, row['Date'], name_stations)
            OK = kringing(data_input_OK)
            point = db.get_table_pollution_by_name_date_gas(name, row['Date'], gas)
            value_kringing, ss = OK.execute("points", np.array(point.geometry.x), np.array(point.geometry.y))
            kringing_values.append(value_kringing.item(0))
            # interpolated
            data = db.get_pollution_interpolated_in_station(name, row['Date'], gas)
            distances = data['distance']
            level_gas = data[gas]
            value_interpolated = float(sum([float(level_gas[i]) / float(distances[i]) for i in range(0, 3)])) / float(
                sum([1.0 / float(distances[i]) for i in range(0, 3)]))
            interpolated_values.append(value_interpolated)
            # model
            input_dataframe = normalize_dataframe_train(group.loc[[index]])
            input_test = transform_dataframe_to_tensor(input_dataframe)
            with torch.no_grad():
                value_model = model(input_test).detach().numpy().item(0)
            model_values.append(value_model)

            # model inter
            input_dataframe = db.get_input_from_interpol(name, row['Date'], gas)
            input_dataframe = normalize_dataframe_train_interpol(input_dataframe, gas)

            with torch.no_grad():
                value_model_I = model_I(input_dataframe).detach().numpy().item(0)
            model_i_values.append(value_model_I)
            # combinated
            value_combinated = (value_model + value_interpolated + value_model_I + value_kringing.item(0)) / 4.0

            combinated_values.append(value_combinated)
            if count > 150:
                break
        real_values = np.array(real_values)
        interpolated_values = np.array(interpolated_values)
        model_values = np.array(model_values)
        model_i_values = np.array(model_i_values)
        kringing_values = np.array(kringing_values)
        combinated_values = np.array(combinated_values)
        error_interpolated = np.sum(np.abs(np.subtract(interpolated_values, real_values))) / len(real_values)
        error_model = np.sum(np.abs(np.subtract(model_values, real_values))) / len(real_values)
        error_kringing = np.sum(np.abs(np.subtract(kringing_values, real_values))) / len(real_values)
        error_combinated = np.sum(np.abs(np.subtract(combinated_values, real_values))) / len(real_values)
        error_i_model = np.sum(np.abs(np.subtract(model_i_values, real_values))) / len(real_values)
        interpolated_sum_error.append(error_interpolated)
        kringing_sum_error.append(error_kringing)
        model_sum_error.append(error_model)
        model_i_sum_error.append(error_i_model)
        combinated_sum_error.append(error_combinated)
        print(
            f'Error en la estacion {name}, interpolacion: {error_interpolated}, modelo: {error_model}, kringing: {error_kringing}, combinados: {error_combinated}, modelo interpolacion: {error_i_model}')
    print(f'Resultados finales, errores:  modelo-> {np.array(model_sum_error).mean()}    '
          f'interpolacion-> {np.array(interpolated_sum_error).mean()}     '
          f'kringing-> {np.array(kringing_sum_error).mean()}    '
          f'combinados-> {np.array(combinated_sum_error).mean()}     '
          f'modelo_interpolacion-> {np.array(model_i_sum_error).mean()}')
