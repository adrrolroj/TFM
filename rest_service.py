import requests
import unidecode
import db_calls as db
import pandas as pd
import time

def localizate(data):
    for i, (index, row) in enumerate(data.iterrows()):
        print(i)
        repeat = True
        while(repeat):
            if str(row['NUM_VIA']) != '-1':
                direction = f"{row['MUNICIPIO']} {row['NOM_VIA']} {row['NUM_VIA']}"
            else:
                direction = f"{row['MUNICIPIO']} {row['NOM_VIA']}"
            url = f'https://eu1.locationiq.com/v1/search.php?key=868225b4fa0e09&q=${direction}&format=json'
            try:
                responses = requests.get(url).json()
                for response in responses:
                    name = unidecode.unidecode(response['display_name'])
                    if unidecode.unidecode(row['PROVINCIA'].split()[-1]) in name:
                        data.at[index, 'COD_LATITUD'] = response['lat']
                        data.at[index, 'COD_LONGITUD'] = response['lon']
                repeat=False
            except Exception as e:
                print(e)
                time.sleep(1)
        #if i%100==0:
        #    db.update_table_to_geohealth(data,'pacientes_modified',geom=False)
    return data


def export_direction_csv(data):
    directions = {'address': [], 'city': [],  'zip': [], 'country': []}
    for index, row in data.iterrows():
        if str(row['NUM_VIA']) != '-1':
            direction = f"{row['NOM_VIA']} {row['NUM_VIA']}"
        else:
            direction = f"{row['NOM_VIA']}"
        city = row['MUNICIPIO'].split("-", 1)[1].rstrip().lstrip()
        directions['address'].append(unidecode.unidecode(direction))
        directions['city'].append(unidecode.unidecode(city))
        directions['zip'].append(row['COD_POSTAL'])
        directions['country'].append('Spain')
        # url = f'https://eu1.locationiq.com/v1/search.php?key=868225b4fa0e09&q=${direction}&format=json'
        '''try:
            responses = requests.get(url).json()
            for response in responses:
                name = unidecode.unidecode(response['display_name'])
                print(name)
                if unidecode.unidecode(row['PROVINCIA'].split()[-1]) in name:
                    data.at[index, 'COD_LATITUD'] = response['lat']
                    data.at[index, 'COD_LONGITUD'] = response['lon']
                    print(data.loc[[index]])
                    break
        except:
            pass'''
    directions = pd.DataFrame.from_dict(directions)
    directions.to_csv('direcciones.csv', index=False)
    return data