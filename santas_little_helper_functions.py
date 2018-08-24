import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from itertools import chain

def compute_weariness(df_gift_trips):

    if np.any(df_gift_trips.groupby('TripId')['Weight'].sum() > 1000):
        print('over limit')

    weariness = 0
    for trip in df_gift_trips['TripId'].unique():
        df_local = df_gift_trips[df_gift_trips.TripId==trip]
        prev_point = (90,0)
        carried_weight = 10 + df_local['Weight'].sum()
        for row in df_local[['Latitude','Longitude','Weight']].values:
            distance = haversine(prev_point[1], prev_point[0], row[1], row[0])
            weariness += distance*carried_weight
            carried_weight -= row[2]
            prev_point = row[0:2]

        distance = haversine(prev_point[1], prev_point[0], 0, 90)
        weariness += distance*carried_weight

    return weariness

# borrowed from stack exchange, takes degrees lat/long and returns distance dimensionless units
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # kilometers
    return c * r

# borrowed from matplotlib documentation
def draw_map(m, scale=0.2):
    m.shadedrelief(scale=scale)
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')



def get_trips(df_region, trip_number, trip_weight):
    df_region_trips = df_region.copy()
    df_region_trips['TripId'] = -1
    #trip_number = 0
    while df_region.shape[0] > 0:
        df_region.reset_index(drop=True, inplace=True)
        center = (df_region['Latitude'].mean(), df_region['Longitude'].mean())
        d_from_center = df_region[['Latitude', 'Longitude']]-center
        seed_point = np.sqrt(d_from_center['Latitude']**2+d_from_center['Longitude']**2).values.argmax()
        seed_location = df_region[['Latitude', 'Longitude']].iloc[seed_point]
        d_from_seed = df_region[['Latitude', 'Longitude']]-seed_location
        d_from_seed = np.sqrt(d_from_seed['Latitude']**2 + d_from_seed['Longitude']**2)
        ordered_dist_from_seed = np.argsort(d_from_seed)
        cumulative_weight = np.cumsum(df_region['Weight'].values[ordered_dist_from_seed])
        trip_mask = cumulative_weight < trip_weight
        gift_ids = df_region.loc[ordered_dist_from_seed[trip_mask].values]['GiftId'].values
        df_region_trips.loc[df_region_trips['GiftId'].isin(gift_ids), 'TripId'] = trip_number
        df_region.drop(ordered_dist_from_seed[trip_mask].values, inplace=True)
        trip_number+= 1

    return df_region_trips, trip_number
