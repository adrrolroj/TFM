import psycopg2 as psycopg2
import geopandas as gpd
import pandas as pd
from dateutil.relativedelta import relativedelta
from geoalchemy2 import WKTElement
from sqlalchemy import create_engine


def get_table(query, data_base):
    conection = psycopg2.connect(database=data_base, user='docker', password='docker',
                                 host='localhost', port='25434')
    try:
        data = gpd.GeoDataFrame.from_postgis(query, conection, geom_col='geometry')
        conection.close()
        return data
    except Exception as ex:
        conection.close()
        print('An exception occurred')
        print(ex)


def get_table_without_geometry(query, data_base):
    engine = create_engine(f'postgresql://docker:docker@localhost:25434/{data_base}', echo=False)
    try:
        data = pd.read_sql_query(query, con=engine)
        return data
    except Exception as ex:
        print('An exception occurred')
        print(ex)


def get_value(query, data_base):
    conection = psycopg2.connect(database=data_base, user='docker', password='docker',
                                 host='localhost', port='25434')
    try:
        cur = conection.cursor()
        cur.execute(query)
        result = cur.fetchone()[0]
        conection.close()
        return result
    except Exception as ex:
        conection.close()
        print('An exception occurred')
        print(ex)


def get_table_censal():
    query = 'SELECT * FROM censal_full'
    return get_table(query, 'GeoHealth')


def get_table_meteo():
    query = 'SELECT * FROM meteo'
    return get_table(query, 'GeoHealth')


def get_geometry_by_cusec(cusec):
    query = f'select t.geometry from censal t where t."CUSEC" =\'{cusec}\''
    return get_table(query, 'GeoHealth')


def get_table_basics_health_region():
    query = 'SELECT * FROM basics_health_region'
    return get_table(query, 'GeoHealth')


def get_table_health_districts_region():
    query = 'SELECT * FROM health_districts_region'
    return get_table(query, 'GeoHealth')


def get_table_postal():
    query = 'SELECT * FROM postal'
    return get_table(query, 'GeoHealth')


def get_table_neighborhood_region():
    query = 'SELECT * FROM neighborhood_region'
    return get_table(query, 'GeoHealth')


def get_avg_atribute_from_neighbor(geometry, atribute):
    atribute = f'"{atribute}"'
    geometry = f"'{geometry}'"
    query = f'SELECT avg(a.{atribute}) from censal_full as a ' \
            f'where ST_Intersects(a.geometry, ST_GeomFromText({geometry}, 4326))'
    return get_value(query, 'GeoHealth')


def get_pollution_xy_time_density_value(atribute, station_names=None):
    if station_names is None:
        query = f'select EXTRACT(DOY FROM p."Date") as doy, EXTRACT(YEAR FROM p."Date") as year, ' \
                f'p."{atribute}" as {atribute}, ST_X(p.geometry) as X,ST_Y(p.geometry) as Y, ' \
                f'p.density_neigh as density_neigh , p.density as density, m."Average_Temperature" as temp' \
                f', m."Average_Wind_Speed" as wind, m."Precipitation" as rain ' \
                f'from  pollution_for_train p ' \
                f'join meteo m on ' \
                f'm.geometry=ST_ClosestPoint((select st_collect(m.geometry) from station_meteo m), p.geometry)' \
                f'where p."{atribute}" is not null and m."Date"=p."Date"'
    else:
        station_names = tuple(station_names)
        query = f'select EXTRACT(DOY FROM p."Date") as doy, EXTRACT(YEAR FROM p."Date") as year, ' \
                f'p."{atribute}" as {atribute}, ST_X(p.geometry) as X,ST_Y(p.geometry) as Y, ' \
                f'p.density_neigh as density_neigh , p.density as density, m."Average_Temperature" as temp,' \
                f' m."Average_Wind_Speed" as wind, m."Precipitation" as rain ' \
                f'from  pollution_for_train p ' \
                f'join meteo m on ' \
                f'm.geometry=ST_ClosestPoint((select st_collect(m.geometry) from station_meteo m), p.geometry)' \
                f'where p."{atribute}" is not null and m."Date"=p."Date" and p."Name" in {station_names}'
    return get_table_without_geometry(query, 'GeoHealth')


def get_table_meteo_by_name(name):
    name = f"'{name}'"
    query = f'select m.* from meteo m where m."Name"={name}'
    return get_table_without_geometry(query, 'GeoHealth')


def get_table_pollution_by_name(name):
    name = f"'{name}'"
    query = f'select m.* from pollution m where m."Name"={name}'
    return get_table_without_geometry(query, 'GeoHealth')


def update_data_for_training():
    query = 'select p.*, c.neigh_density as density_neigh, c.pob8 as density from pollution p ' \
            'join censal_full c on ST_Contains(c.geometry, p.geometry)'
    data = get_table(query, 'GeoHealth')
    update_table_to_geohealth(data, 'pollution_for_train')


def get_center_density_censal(date_from):
    date_from = f"'{date_from}'"
    query = f'select ST_X(ST_Centroid(c.geometry)) as X, ST_Y(ST_Centroid(c.geometry)) as Y, c.neigh_density ' \
            f'as density_neigh, c.pob8 as density, m."Average_Wind_Speed" as wind, m."Precipitation" as rain, ' \
            f'm."Average_Temperature" as temp, c.cusec as cusec ' \
            f'from censal_full c left join meteo m on m.geometry=ST_ClosestPoint((select st_collect(m.geometry) ' \
            f'from station_meteo m), ST_Centroid(c.geometry)) where m."Date"={date_from}'
    return get_table_without_geometry(query, 'GeoHealth')


def get_min_max_censal():
    query = f'select max(c.neigh_density) as max_dn, min(c.neigh_density) as min_dn, ' \
            f'max(c.pob8) as max_d, min(c.pob8) as min_d, ' \
            f'max(ST_XMax(c.geometry)) as max_x, min(ST_XMin(c.geometry)) as min_x,' \
            f' max(ST_YMax(c.geometry)) as max_y, min(ST_YMin(c.geometry)) as min_y from censal_full c'
    query_2 = f'select  max(m."Average_Wind_Speed") as max_wind, max(m."Precipitation") as max_rain, ' \
              f'min(m."Average_Temperature") as min_temp, max(m."Average_Temperature") as max_temp from meteo m'
    data = get_table_without_geometry(query, 'GeoHealth')
    data = pd.concat([data, get_table_without_geometry(query_2, 'GeoHealth')], axis=1)
    return data


def get_xy_density_from_grid_points():
    query = f'select ST_X(p.geometry) as X,ST_Y(p.geometry) as Y, c.neigh_density as density ' \
            f'from  grid_points p join censal_full c on st_within( p.geometry, c.geometry) '
    return get_table_without_geometry(query, 'GeoHealth')


def get_avg_atribute(atribute):
    atribute = f'"{atribute}"'
    query = f'SELECT avg(a.{atribute}) from censal_full as a'
    return get_value(query, 'GeoHealth')


def get_table_pollution():
    query = 'SELECT * FROM pollution'
    return get_table(query, 'GeoHealth')


def get_table_pollution_from_dates(date_from, date_to):
    query = f'SELECT p."Name", p."Province", p."Municipality",' \
            f'avg(p."SH2") as "SH2", avg(p."SO2") as "SO2", ' \
            f'avg(p."NO2") as "NO2", avg(p."CO") as "CO", avg(p."PART") as "PART", ' \
            f'avg(p."O3") as "O3", p.geometry from pollution p' \
            f' where (p."Date" BETWEEN \'{date_from}\' AND \'{date_to}\') ' \
            f'GROUP BY p."Name", p."Province", p."Municipality", p.geometry'
    return get_table(query, 'GeoHealth')


def get_mean_meteo_from_date(atribute, name, date_from, date_to):
    name = f"'{name}'"
    date_from = f"'{date_from}'"
    date_to = f"'{date_to}'"
    query = f'select avg(m."{atribute}") from meteo m where m."Name"={name} ' \
            f'and (m."Date"={date_from} or m."Date"={date_to})'
    return get_value(query, 'GeoHealth')


def get_pollution_from_analysis_group_by_station():
    query = f'SELECT p."Name", p."Province", p."Municipality", avg(p."SH2") as "SH2", avg(p."SO2") as "SO2",' \
            f' avg(p."NO2") as "NO2", avg(p."CO") as "CO", avg(p."PART") as "PART", avg(p."O3") as "O3", ' \
            f'p.geometry from pollution_in_month p GROUP BY p."Name", p."Province", p."Municipality", p.geometry'
    return get_table(query, 'Analysis')


def get_table_pollution_from_analysis():
    query = 'SELECT * FROM pollution_in_month'
    return get_table(query, 'Analysis')


def get_table_grid_from_analysis():
    query = 'SELECT * FROM points_grid'
    return get_table(query, 'Analysis')


def get_table_analysis_by_CUSEC(table, cusec):
    query = f'SELECT * FROM {table} t where t."cusec" =\'{cusec}\''
    return get_table(query, 'Analysis')


def compare_cluster_from_analysis(cluster1, cluster2, segmentation):
    query = f'select a.{segmentation} as first, b.{segmentation} as second,' \
            f' count(a.{segmentation}) from {cluster1} a join {cluster2} b ' \
            f'on a.geometry = b.geometry group by a.{segmentation}, b.{segmentation}'
    return get_table_without_geometry(query, 'Analysis')


def get_station_pollution_names():
    query = 'select m."Name" as name from pollution m group by m."Name"'
    return get_table_without_geometry(query, 'GeoHealth')


def get_station_pollution_names_if_not_empty(atribute):
    query = f'select m."Name" as name from pollution m where m."{atribute}" is not null group by m."Name"'
    return get_table_without_geometry(query, 'GeoHealth')


def get_pollution_intersect(date, name, atribute):
    atribute = f'"{atribute}"'
    date_from = date - relativedelta(months=1)
    date_to = date + relativedelta(months=1)

    query = f'SELECT avg(p.{atribute}) from pollution_in_month p' \
            f' where (p."Date" BETWEEN \'{date_from}\' AND \'{date_to}\') and p."Name" = \'{name}\''
    return get_value(query, 'Analysis')


def make_grid(meters):  # Crea un cuadrado de puntos en andalucia con una distancia dada
    conection = psycopg2.connect(database='GeoHealth', user='docker', password='docker',
                                 host='localhost', port='25434')
    query = f'select st_transform(makegrid(st_transform(ST_Union(ST_SnapToGrid(a.geometry,0.000001)), 2062),' \
            f' {meters}), 4326) as geometry from health_districts_region a'
    try:
        data = gpd.GeoDataFrame.from_postgis(query, conection, geom_col='geometry')
        data = data.explode(ignore_index=True)
        conection.close()
        return data
    except Exception as ex:
        conection.close()
        print('An exception occurred')
        print(ex)


def join_grid_with_censal(atribute):
    query = f'select c.*, t."{atribute}" from censal_full c join temp_grid t on t.cusec=c.cusec'
    return get_table(query, 'Analysis')


def update_table_to_analysis(data, name_table, geom=True):
    update_table(data, name_table, 'Analysis', geom)


def update_table_to_geohealth(data, name_table, geom=True):
    update_table(data, name_table, 'GeoHealth', geom)


def update_table(data, name_table, database, geom=True):
    engine = create_engine(f'postgresql://docker:docker@localhost:25434/{database}', echo=False)
    if geom:
        data['geometry'] = data['geometry'].apply(
            lambda geo: str(WKTElement(geo.wkt, srid=4326)) if not type(geo) == str else geo)
    try:
        data.to_sql(name_table, engine, if_exists='replace', index=False)
        if geom:
            with engine.connect() as conn, conn.begin():
                sql = f'ALTER TABLE {name_table} ALTER COLUMN geometry TYPE Geometry ' \
                      f'USING ST_SetSRID(geometry::Geometry, 4326)'
                conn.execute(sql)
        print(f'Table {name_table} updated succesfully!')
    except Exception as ex:
        print('An exception occurred')
        print(ex)


def drop_table_analysis(name_table):
    conection = psycopg2.connect(database='Analysis', user='docker', password='docker',
                                 host='localhost', port='25434')
    query = f'DROP TABLE IF EXISTS {name_table}'
    try:
        cur = conection.cursor()
        cur.execute(query)
        conection.close()
    except Exception as ex:
        conection.close()
        print('An exception occurred')
        print(ex)


#####################################
############# GEOASMA ###############
#####################################

def update_table_GeoAsma(data, name_table, geom=True):
    engine = create_engine(f'postgresql://docker:docker@10.232.5.113:25434/GeoAsma', echo=False)
    if geom:
        data['geometry'] = data['geometry'].apply(
            lambda geom: str(WKTElement(geom.wkt, srid=4326)) if not type(geom) == str else geom)
    try:
        data.to_sql(name_table, engine, if_exists='replace', index=False)
        if geom:
            with engine.connect() as conn, conn.begin():
                sql = f'ALTER TABLE {name_table} ALTER COLUMN geometry TYPE Geometry ' \
                      f'USING ST_SetSRID(geometry::Geometry, 4326)'
                conn.execute(sql)
        print(f'Table {name_table} updated successfully!')
    except Exception as ex:
        print('An exception occurred')
        print(ex)


def get_table_geoasma(query):
    engine = create_engine(f'postgresql://docker:docker@10.232.5.113:25434/GeoAsma', echo=False)
    try:
        data = pd.read_sql_query(query, con=engine)
        return data
    except Exception as ex:
        print('An exception occurred')
        print(ex)
