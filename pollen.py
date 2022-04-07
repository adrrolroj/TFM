import asyncio
import datetime
import json
import pandas as pd
import websockets
from shapely.geometry import Point
import db_calls as db


def process_data_pollen(data,
                        update=True):  # Procesa los datos de polen recibidos como entrada para subirlo a la base de datos GeoHealth
    result = pd.DataFrame()
    info = {'id': ['4', '22', '23', '30', '41', '47', '58', '60', '62', '65'],
            'name': ['Almeria', 'Jaen', 'Jaen-Hospital', 'Malaga', 'Sevilla-Macarena', 'Sevilla-Tomillar', 'Cadiz',
                     'Huelva', 'Cordoba', 'Granada'],
            'province': ['Almeria', 'Jaen', 'Jaen', 'Malaga', 'Sevilla', 'Sevilla', 'Cadiz', 'Huelva', 'Cordoba',
                         'Granada'],
            'ccaa': ['Andalucia', 'Andalucia', 'Andalucia', 'Andalucia', 'Andalucia', 'Andalucia', 'Andalucia',
                     'Andalucia', 'Andalucia', 'Andalucia'],
            'geometry': [Point(-2.4596435389546327, 36.847114768612194),
                         Point(-3.7798110193798204, 37.79649750505875),
                         Point(-3.7942113706324627, 37.77617996718808),
                         Point(-4.431633598417232, 36.72698443725751),
                         Point(-5.987224074598175, 37.40629267557687),
                         Point(-5.901279099222474, 37.2985901612703),
                         Point(-6.277720859801188, 36.508570296898256),
                         Point(-6.926413589932919, 37.27899689683054),
                         Point(-4.798242228848024, 37.86846639203616),
                         Point(-3.6044834663997625, 37.18624002257025)]}
    for index, row in data.iterrows():
        line = {}
        i = info['id'].index(str(row['idEstacion']))
        line['id'] = row['idEstacion']
        line['name'] = info['name'][i]
        line['province'] = info['province'][i]
        line['ccaa'] = 'Andalucia'
        line['geometry'] = str(info['geometry'][i])
        fecha = str(row['fecha'])
        date_time_str = f'{fecha[0:4]}-{fecha[4:6]}-{fecha[6:8]} 00:00:00.000000'
        line['date'] = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')
        for j in range(1, 23):
            line[f'{j}'] = row[f'{j}']
        line = pd.DataFrame([line])
        line.rename(columns={'idEstacion': 'id', 'fecha': 'date', '1': 'Alnus', '2': 'Alternaria', '3': 'Artemisa',
                             '4': 'Betula',
                             '5': 'Carex', '6': 'Castanea', '7': 'Cupresaceas', '8': 'Fraxinus', '9': 'Gramineas',
                             '10': 'Mercurialis',
                             '11': 'Morus', '12': 'Olea', '13': 'Palmaceas', '14': 'Pinus', '15': 'Plantago',
                             '16': 'Platanus', '17': 'Populus',
                             '18': 'Amarantaceas', '19': 'Quercus', '20': 'Rumex', '21': 'Ulmus', '22': 'Urticaceas'},
                    inplace=True)
        result = pd.concat([result, line])
    if update:
        db.update_table_to_geohealth(result, 'pollen_full', geom=False)
        return result
    else:
        return result


@asyncio.coroutine
def get_data_pollen(from_date, to_date,
                    update=True):  # Obtiene los datos de polen para las estaciones a las que le indico por las ids
    ids = ['4', '22', '23', '30', '41', '47', '58', '60', '62', '65']
    data = pd.DataFrame()
    for id in ids:
        print(f'Para la id {id}')
        websocket = yield from websockets.connect('wss://www.polenes.com/sockjs/045/4ec9at4_/websocket')
        name = '[\"{\\\"msg\\\":\\\"sub\\\",\\\"id\\\":\\\"65qZKBe8CjpxJkiK4\\\",\\\"name\\\":\\\"polenes2015\\\",\\\"params\\\":[{\\\"selector\\\":{\\\"$and\\\":[{\\\"fecha\\\":{\\\"$gte\\\":' + str(
            from_date) + ',\\\"$lte\\\":' + str(to_date) + '}},{\\\"idEstacion\\\":' + str(
            id) + '}]},\\\"options\\\":{\\\"sort\\\":{\\\"fecha\\\":1}},\\\"jump\\\":1}]}\"]'
        yield from websocket.send(
            '[\"{\\\"msg\\\":\\\"connect\\\",\\\"version\\\":\\\"1\\\",\\\"support\\\":[\\\"1\\\",\\\"pre2\\\",\\\"pre1\\\"]}\"]')
        yield from websocket.send(
            '[\"{\\\"msg\\\":\\\"sub\\\",\\\"id\\\":\\\"TcAtwX89rKhgG6LQo\\\",\\\"name\\\":\\\"meteor.loginServiceConfiguration\\\",\\\"params\\\":[]}\"]')
        yield from websocket.send(name)
        counter = 0
        save = []
        print('Recibiendo datos...')
        while (True):
            try:
                greeting = yield from asyncio.wait_for(websocket.recv(), 3)
                if counter > 4:
                    save.append(str(greeting)[3:-2])
                counter += 1
            except:
                break
        yield from websocket.close()
        print('Datos recibidos procensando...')
        for line in save:
            line = line.replace('\\', '')
            dictionary = json.loads(line)
            if dictionary['msg'] == 'added':
                fields = dictionary['fields']
                dictionary['fields'] = None
                dictionary = pd.DataFrame([dictionary])
                fields = pd.DataFrame([fields])
                dictionary = pd.concat([dictionary, fields], axis=1)
                data = pd.concat([data, dictionary])
            else:
                break
        print('Procesamiento completado')
    print('Finalizado')
    result = process_data_pollen(data, update)
    return result


def update_data_pollen():
    data = db.get_table_pollen()
    date_max = max(data['date']).strftime('%Y%m%d')
    today = datetime.date.today().strftime('%Y%m%d')
    data_to_add = asyncio.get_event_loop().run_until_complete(get_data_pollen(date_max, today, update=False))
    data = pd.concat([data, data_to_add])
    data = data.drop_duplicates(subset=['date', 'id'])
    db.update_table_to_geohealth(data, 'pollen', geom=True)


update_data_pollen()
