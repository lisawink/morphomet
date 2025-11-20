import pandas as pd
import geopandas as gpd
import numpy as np
import momepy
import libpysal
import geoplanar
from itertools import combinations
from shapely.geometry import Point, Polygon
from scipy.stats import iqr
from scipy.stats import median_abs_deviation
from scipy.stats import skew
import rasterio
from rasterio import mask
from rasterstats import zonal_stats
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import statsmodels.api as sm

def convert_to_point(stations, input_crs='EPSG:4326', output_crs='EPSG:31468', lat_column = 'station_lat', lon_column = 'station_lon'):
    """
    Converts a DataFrame of station coordinates to a GeoDataFrame with Point geometries
    Parameters
    ----------
    stations : DataFrame
        DataFrame containing station coordinates
    input_crs : str, optional
        Input coordinate reference system (CRS) of the DataFrame. The default is 'EPSG:4326'.
    output_crs : str, optional
        Output coordinate reference system (CRS) for the GeoDataFrame. The default is 'EPSG:31468'.
    lat_column : str, optional
        Name of the column containing latitude values. The default is 'station_lat'.
    lon_column : str, optional
        Name of the column containing longitude values. The default is 'station_lon'.
    Returns
    -------
    GeoDataFrame
        A GeoDataFrame with Point geometries representing the station coordinates, transformed to the specified output CRS.
    """    
    geometry = [Point(xy) for xy in zip(stations[lon_column], stations[lat_column])]
    stn_gdf = gpd.GeoDataFrame(stations, crs=input_crs, geometry=geometry)
    stn_gdf = stn_gdf.to_crs(output_crs)
    return stn_gdf

def buffer_stations(stations, radius=100, input_crs='EPSG:4326', output_crs='EPSG:31468', lat_column = 'station_lat', lon_column = 'station_lon'):
    """
    Converts a DataFrame of station coordinates to a GeoDataFrame with buffer geometries
    Parameters
    ----------
    stations : DataFrame
        DataFrame containing station coordinates
    radius : float, optional
        Radius of the buffer in the units of the output CRS. The default is 100.
    input_crs : str, optional
        Input coordinate reference system (CRS) of the DataFrame. The default is 'EPSG:4326'.
    output_crs : str, optional
        Output coordinate reference system (CRS) for the GeoDataFrame. The default is 'EPSG:31468'.
    lat_column : str, optional
        Name of the column containing latitude values. The default is 'station_lat'.
    lon_column : str, optional
        Name of the column containing longitude values. The default is 'station_lon'.
    Returns
    -------
    GeoDataFrame
        A GeoDataFrame with buffer geometries representing the station coordinates, transformed to the specified output CRS.
        """
    geometry = [Point(xy) for xy in zip(stations[lon_column], stations[lat_column])]
    stn_gdf = gpd.GeoDataFrame(stations, crs=input_crs, geometry=geometry)
    stn_gdf = stn_gdf.to_crs(output_crs)
    stn_gdf['geometry'] = stn_gdf.buffer(radius)
    return stn_gdf

def city_centre_distance(stations, city_centre, input_crs='EPSG:4326', output_crs='EPSG:31468', lat_column = 'station_lat', lon_column = 'station_lon'):
    """
    Calculates the distance from each station to the city centre
    Parameters
    ----------
    stations : DataFrame
        DataFrame containing station coordinates
    city_centre : shapely Point
        Point representing the city centre
    input_crs : str, optional
        Input coordinate reference system (CRS) of the DataFrame. The default is 'EPSG:4326'.
    output_crs : str, optional
        Output coordinate reference system (CRS) for the GeoDataFrame. The default is 'EPSG:31468'.
    lat_column : str, optional
        Name of the column containing latitude values. The default is 'station_lat'.
    lon_column : str, optional
        Name of the column containing longitude values. The default is 'station_lon'.
    Returns
    -------
    GeoDataFrame
        A GeoDataFrame with Point geometries representing the station coordinates, transformed to the specified output CRS, and a new column 'city_centre_distance' containing the distance to the city centre.
        """
    geometry = [Point(xy) for xy in zip(stations[lon_column], stations[lat_column])]
    stn_gdf = gpd.GeoDataFrame(stations, crs=input_crs, geometry=geometry)
    stn_gdf = stn_gdf.to_crs(output_crs)
    stn_gdf['city_centre_distance'] = stn_gdf.distance(city_centre)
    return stn_gdf

def random_buffers(buildings, number=50, radius=100):
    """
    Generates buffers of set radius around random buildings
    Parameters
    ----------
    buildings : GeoDataFrame
        GeoDataFrame containing building footprints
    number : int, optional
        Number of random building buffers to generate. The default is 50.
    radius : float, optional
        Radius of the buffer in the units of the buildings GeoDataFrame CRS. The default is 100.
    Returns
    -------
    GeoDataFrame
        A GeoDataFrame with buffer geometries representing random buildings.
    """
    random_building_buffers = buildings.sample(n=number)
    random_building_buffers['geometry'] = random_building_buffers.centroid.buffer(radius)
    random_building_buffers.set_geometry('geometry', inplace=True)
    return random_building_buffers

def make_compass_pie_slice(center, radius, compass_center_angle_deg, angle_width_deg, num_arc_points=30):
    """
    Create a pie slice centered on a compass angle (0° = North, 90° = East, etc.).
    
    Parameters:
        center: shapely Point
        radius: radius of the slice
        compass_center_angle_deg: compass angle (0° = North, increases clockwise)
        angle_width_deg: angular width of the slice (in degrees)
        num_arc_points: number of arc points for smoothness
        
    Returns:
        shapely Polygon representing the pie slice
    """
    # Compute start and end angles in compass degrees
    half_width = angle_width_deg / 2
    start_compass = compass_center_angle_deg - half_width
    end_compass = compass_center_angle_deg + half_width

    # Convert to mathematical angles (0° = East, CCW increase)
    start_math = np.radians(90 - start_compass)
    end_math = np.radians(90 - end_compass)

    # Create arc points from start to end (CW on compass = CCW in math)
    angles = np.linspace(start_math, end_math, num_arc_points)
    arc = [(center.x + radius * np.cos(a), center.y + radius * np.sin(a)) for a in angles]

    # Construct the polygon (sector)
    coords = [center.coords[0]] + arc + [center.coords[0]]
    return Polygon(coords)

def convert_to_pie(stations, input_crs='EPSG:4326', output_crs='EPSG:31468', lat_column = 'station_lat', lon_column = 'station_lon', radius=100, compass_angle=0, angle_width=60):
    """
    Converts a DataFrame of station coordinates to a GeoDataFrame with pie slice geometries
    Parameters
    ----------
    stations : DataFrame
        DataFrame containing station coordinates
    input_crs : str, optional
        Input coordinate reference system (CRS) of the DataFrame. The default is 'EPSG:4326'.
    output_crs : str, optional
        Output coordinate reference system (CRS) for the GeoDataFrame. The default is 'EPSG:31468'.
    lat_column : str, optional
        Name of the column containing latitude values. The default is 'station_lat'.
    lon_column : str, optional
        Name of the column containing longitude values. The default is 'station_lon'.
    radius : float, optional
        Radius of the pie slice in the units of the output CRS. The default is 100.
    compass_angle : float, optional
        Compass angle (0° = North, 90° = East, etc.) for the center of the pie slice. The default is 0.
    angle_width : float, optional
        Angular width of the pie slice (in degrees). The default is 60.
    Returns
    -------
    GeoDataFrame
        A GeoDataFrame with pie slice geometries representing the station coordinates, transformed to the specified output CRS.
        """
    geometry = [Point(xy) for xy in zip(stations[lon_column], stations[lat_column])]
    stn_gdf = gpd.GeoDataFrame(stations, crs=input_crs, geometry=geometry)
    stn_gdf = stn_gdf.to_crs(output_crs)
    sectors = [make_compass_pie_slice(geom, radius, compass_angle, angle_width) for geom in stn_gdf.geometry]
    stn_gdf = gpd.GeoDataFrame(stations, crs=output_crs, geometry=sectors)
    return stn_gdf

def block_params(buildings,height,streets):

    """
    Extracts the following parameters for each building and street segment in the city:
    - dimension
    - courtyards
    - shape
    - proximity
    - streets
    - intensity
    - connectivity
    - COINS

    Parameters
    ----------
    buildings : GeoDataFrame
        GeoDataFrame containing building footprints
    height : float
        Series containing building heights
    streets : GeoDataFrame
        GeoDataFrame containing street segments

    Returns
    -------
    bldgs : GeoDataFrame
        GeoDataFrame containing building parameters
    streets : GeoDataFrame
        GeoDataFrame containing street parameters
    nodes : GeoDataFrame
        GeoDataFrame containing nodes
    edges : GeoDataFrame
        GeoDataFrame containing edges

    """

    bldgs = buildings.copy()

    # dimension
    bldgs['BuAre'] = bldgs.geometry.area
    bldgs['BuHt'] = height
    bldgs['BuPer'] = bldgs.geometry.length
    bldgs['BuLAL'] = momepy.longest_axis_length(bldgs)
    bldgs[['BuCCD_mean','BuCCD_std']] = momepy.centroid_corner_distance(bldgs)
    bldgs['BuCor'] = momepy.corners(bldgs)

    # courtyards
    bldgs['CyAre'] = momepy.courtyard_area(bldgs)
    bldgs['CyInd'] = momepy.courtyard_index(bldgs)

    # shape
    bldgs['BuCCo'] = momepy.circular_compactness(bldgs)
    bldgs['BuCWA'] = momepy.compactness_weighted_axis(bldgs, bldgs['BuLAL'])
    bldgs['BuCon'] = momepy.convexity(bldgs)
    bldgs['BuElo'] = momepy.elongation(bldgs)
    bldgs['BuERI'] = momepy.equivalent_rectangular_index(bldgs)
    bldgs['BuFR'] = momepy.facade_ratio(bldgs)
    bldgs['BuFF'] = momepy.form_factor(bldgs, height)
    bldgs['BuFD'] = momepy.fractal_dimension(bldgs)
    bldgs['BuRec'] = momepy.rectangularity(bldgs)
    bldgs['BuShI'] = momepy.shape_index(bldgs, bldgs['BuLAL'])
    bldgs['BuSqC'] = momepy.square_compactness(bldgs)
    bldgs['BuCorDev'] = momepy.squareness(bldgs)

    # proximity
    bldgs['BuSW'] = momepy.shared_walls(bldgs)
    bldgs['BuSWR'] = momepy.shared_walls(bldgs)/bldgs['BuPer']
    bldgs['BuOri'] = momepy.orientation(bldgs)

    ## building adjacency

    delaunay = libpysal.graph.Graph.build_triangulation(geoplanar.trim_overlaps(bldgs).centroid).assign_self_weight()
    bldgs['BuAli'] = momepy.alignment(bldgs['BuOri'], delaunay)

    # streets
    bldgs["street_index"] = momepy.get_nearest_street(bldgs, streets)
    streets['StrLen'] = streets.geometry.length

    # street alignment
    str_orient = momepy.orientation(streets)
    bldgs['StrAli'] = momepy.street_alignment(momepy.orientation(bldgs), str_orient, bldgs["street_index"])

    # street profile
    streets[['StrW','StrOpe','StrWD','StrH','StrHD','StrHW']] = momepy.street_profile(streets, bldgs, height=height)

    # intensity
    building_count = momepy.describe_agg(bldgs['BuAre'], bldgs["street_index"], statistics=["count"])
    streets = streets.merge(building_count, left_on=streets.index, right_on='street_index', how='left')
    streets['BpM'] = streets['count'] / streets.length

    #shape
    streets['StrLin'] = momepy.linearity(streets)

    #connectivity
 
    graph = momepy.gdf_to_nx(streets)
    graph = momepy.closeness_centrality(graph, radius=400, name="StrClo400", distance="mm_len", weight="mm_len")
    #graph = momepy.closeness_centrality(graph, radius=1200, name="StrClo1200", distance="mm_len", weight="mm_len")
    graph = momepy.betweenness_centrality(graph, radius=400, name="StrBet400", distance="mm_len", weight="mm_len")
    #graph = momepy.betweenness_centrality(graph, radius=1200, name="StrBet1200", distance="mm_len", weight="mm_len")
    graph = momepy.meshedness(graph, radius=400, distance="mm_len", name="StrMes400")
    #graph = momepy.meshedness(graph, radius=1200, distance="mm_len", name="StrMes1200")
    graph = momepy.gamma(graph, radius=400, distance="mm_len", name="StrGam400")
    #graph = momepy.gamma(graph, radius=1200, distance="mm_len", name="StrGam1200")
    graph = momepy.cyclomatic(graph, radius=400, distance="mm_len", name="StrCyc400")
    #graph = momepy.cyclomatic(graph, radius=1200, distance="mm_len", name="StrCyc1200")
    graph = momepy.edge_node_ratio(graph, radius=400, distance="mm_len", name="StrENR400")
    #graph = momepy.edge_node_ratio(graph, radius=1200, distance="mm_len", name="StrENR1200")
    graph = momepy.node_degree(graph, name='StrDeg')
    graph = momepy.clustering(graph, name='StrSCl')
    #graph = momepy.betweenness_centrality(graph, name="StrBetGlo", mode="nodes", weight="mm_len") # will take ages
    nodes, edges = momepy.nx_to_gdf(graph)

    #COINS
    coins = momepy.COINS(streets)
    stroke_gdf = coins.stroke_gdf()
    stroke_attr = coins.stroke_attribute()
    streets['COINS_index'] = stroke_attr
    streets = streets.merge(stroke_gdf, left_on='COINS_index', right_on='stroke_group')
    streets['StrCNS']=streets['geometry_y'].length

    return bldgs, streets, nodes

def neighbourhood_graph_params(buildings, stations):
    """
    
    Extracts the following parameters for each station:
    - Building adjacency
    - Interbuilding distance
        
    Parameters
    ----------
    buildings : GeoDataFrame
        GeoDataFrame containing building footprints
        stations : GeoDataFrame
        GeoDataFrame containing station buffers

    Returns
    -------
    stations : GeoDataFrame
        GeoDataFrame containing station parameters
    
    """
    if 'station_id' not in stations.columns:
        stations['station_id'] = stations.index
    
    buildings = geoplanar.trim_overlaps(buildings)
    overlapping = buildings.sjoin(stations,predicate='within',how='inner')
    
    # create libpysal graph of the buildings in overlapping for each station id
    libpysal_graphs = {}
    
    for stn_id in overlapping['station_id'].unique():

        ol_buildings = overlapping[overlapping['station_id'] == stn_id]
        
        # Generate all unique pairs of indices as adjacency list
        adjacency_list = [(i, j) for i, j in combinations(ol_buildings.index, 2)]

        # Add symmetric pairs to make it undirected (i.e., (i, j) and (j, i))
        adjacency_list += [(j, i) for i, j in adjacency_list]

        # Create a DataFrame from the adjacency list
        adjacency_df = pd.DataFrame(adjacency_list, columns=['focal', 'neighbor'])
        adjacency_df['weight'] = 1  # Assign a default weight of 1

        ref_area_graph = libpysal.graph.Graph.from_adjacency(adjacency_df)

        libpysal_graphs[stn_id] = ref_area_graph

        #calculate bua
        contig = libpysal.graph.Graph.build_contiguity(ol_buildings)
        bua = momepy.building_adjacency(contig, ref_area_graph)
        ol_buildings['BuAdj'] = bua

        #calculate ibd
        if len(ol_buildings) <= 2:
            ol_buildings['BuIBD'] = None
        else:
            delaunay = libpysal.graph.Graph.build_triangulation(ol_buildings.centroid).assign_self_weight()
            ibd = momepy.mean_interbuilding_distance(ol_buildings, delaunay, ref_area_graph)
            ol_buildings['BuIBD'] = ibd

        overlapping.loc[ol_buildings.index, ['BuAdj', 'BuIBD']] = ol_buildings[['BuAdj', 'BuIBD']]

    if 'BuAdj' not in overlapping.columns:
        overlapping['BuAdj'] = None
    if 'BuIBD' not in overlapping.columns:
        overlapping['BuIBD'] = None

    bua = overlapping.groupby('station_id')['BuAdj'].mean()
    ibd = overlapping.groupby('station_id')['BuIBD'].mean()
    stations = stations.merge(bua, left_on='station_id', right_on=bua.index, how='left')
    stations = stations.merge(ibd, left_on='station_id', right_on=ibd.index, how='left')
    
    return stations

def select_objects(buildings, streets, nodes, stations):
    """

    Selects the buildings, streets and nodes for each station


    Parameters
    ----------
    buildings : GeoDataFrame
        GeoDataFrame containing building parameters
    streets : GeoDataFrame
        GeoDataFrame containing street parameters
    nodes : GeoDataFrame
        GeoDataFrame containing node parameters
    stations : GeoDataFrame
        GeoDataFrame containing station buffers

    Returns
    -------
    df : DataFrame
        DataFrame containing aggregated parameters for each station

    """

    # select buildings whose area is at least 50% within the station buffer

    # Calculate the area of each building
    buildings['area'] = buildings.geometry.area
    streets['length'] = streets.geometry.length

    # Perform a spatial join to find buildings that intersect with station buffers
    joined_buildings = gpd.sjoin(buildings, stations, how='inner', predicate='intersects')
    joined_streets = gpd.sjoin(streets, stations, how='inner', predicate='intersects')
    joined_nodes = gpd.sjoin(nodes, stations, how='inner', predicate='intersects')

    # Ensure geodataframes have a column named geometry
    joined_buildings['geometry'] = joined_buildings.geometry
    joined_streets['geometry'] = joined_streets.geometry
    joined_nodes['geometry'] = joined_nodes.geometry

    # Calculate the intersection area for each building-station pair
    intersection_area = joined_buildings.apply(
        lambda row: row.geometry.intersection(stations.loc[row['index_right']].geometry).area, axis=1
    )
    if intersection_area.empty: 
        joined_buildings['intersection_area'] = None
    else:
        joined_buildings['intersection_area'] = intersection_area

    # Calculate the intersection area for each building-station pair
    intersection_length = joined_streets.apply(
        lambda row: row.geometry.intersection(stations.loc[row['index_right']].geometry).length, axis=1
    )
    if intersection_length.empty:
        joined_streets['intersection_length'] = None
    else:
        joined_streets['intersection_length'] = intersection_length

    # Calculate the percentage of each building's area that is within each station buffer
    joined_buildings['percentage_within_buffer'] = (joined_buildings['intersection_area'] / joined_buildings.geometry.area) * 100
    joined_streets['percentage_within_buffer'] = (joined_streets['intersection_length'] / joined_streets.geometry.length) * 100

    # Select buildings where this percentage is at least 50%
    selected_buildings = joined_buildings[joined_buildings['percentage_within_buffer'] >= 50]
    selected_streets = joined_streets[joined_streets['percentage_within_buffer'] >= 50]
    selected_nodes = joined_nodes

    # Output the selected buildings
    selected_buildings = selected_buildings.drop(columns=['index_right'])
    selected_streets = selected_streets.drop(columns=['index_right'])
    
    return selected_buildings, selected_streets, selected_nodes

def weighted_stats(group, i, weight):
    """
    Calculate weighted statistics for a given group.
    Parameters
    ----------
    group : DataFrame
        DataFrame containing the group of data
        i : str
        Column name for which to calculate statistics
        weight : str
        Column name for weights
        Returns
        -------
        Series
        Series containing weighted statistics
        
        """
    weighted_mean = np.sum(group[i] * group[weight]) / np.sum(group[weight])
    weighted_std = np.sqrt(np.sum(group[weight] * (group[i] - weighted_mean) ** 2) / np.sum(group[weight]))

    # weighted median
    sorted_group = group.sort_values(i)
    cumulative_weight = sorted_group[weight].cumsum()
    cutoff = sorted_group[weight].sum() / 2
    weighted_median = sorted_group.loc[cumulative_weight >= cutoff, i].iloc[0]

    # Weighted minimum and maximum
    weighted_min = sorted_group.loc[sorted_group[weight].idxmin(), i]
    weighted_max = sorted_group.loc[sorted_group[weight].idxmax(), i]

    # Weighted sum
    weighted_sum = np.sum(group[i] * group[weight])

    # Weighted mode (most frequently occurring value by weight)
    #mode_idx = group.groupby(i)[weight].sum().idxmax()
    #weighted_mode = mode_idx

    # Weighted 25th and 75th percentiles
    q25_cutoff = sorted_group[weight].sum() * 0.25
    q75_cutoff = sorted_group[weight].sum() * 0.75

    weighted_q25 = sorted_group.loc[cumulative_weight >= q25_cutoff, i].iloc[0]
    weighted_q75 = sorted_group.loc[cumulative_weight >= q75_cutoff, i].iloc[0]

    return pd.Series({
        'weighted_mean': weighted_mean,
        'weighted_std': weighted_std,
        'weighted_median': weighted_median,
        'weighted_min': weighted_min,
        'weighted_max': weighted_max,
        'weighted_sum': weighted_sum,
        'weighted_25th_percentile': weighted_q25,
        'weighted_75th_percentile': weighted_q75,
    })


def aggregate_params(selected_buildings, selected_streets, selected_nodes, stations, weight='BuAre', three_d=False):
    """
    Aggregates the following parameters for each station:
    - Building parameters
    - Street parameters
    - Node parameters
    Parameters
    ----------
    selected_buildings : GeoDataFrame
        GeoDataFrame containing selected building parameters
        selected_streets : GeoDataFrame
        GeoDataFrame containing selected street parameters
        selected_nodes : GeoDataFrame
        GeoDataFrame containing selected node parameters
        stations : GeoDataFrame
        GeoDataFrame containing station buffers
        weight : str, optional
        Weighting parameter for weighted statistics. The default is 'BuAre'.
        three_d : bool, optional
        If True, 3D building parameters are included. The default is False.
        Returns
        -------
        stations : GeoDataFrame
        GeoDataFrame containing aggregated station parameters
        """

    df = pd.DataFrame()

    two_d_list = ['BuAre','BuHt','BuPer','BuLAL','BuCCD_mean','BuCCD_std','BuCor','CyAre','CyInd','BuCCo','BuCWA','BuCon','BuElo','BuERI','BuFR','BuFF','BuFD','BuRec','BuShI','BuSqC','BuCorDev','BuSWR','BuOri','BuAli','StrAli']
    three_d_list = two_d_list + ['BuCir', 'BuHem_3D', 'BuCon_3D', 'BuFra', 'BuFra_3D', 'BuCubo_3D', 'BuSqu', 'BuCube_3D', 'BumVE_3D', 'BuMVE_3D', 'BuFF_3D', 'BuEPI_3D', 'BuProx', 'BuProx_3D', 'BuEx', 'BuEx_3D', 'BuSpi', 'BuSpi_3D', 'BuPerC', 
              'BuCf_3D', 'BuDep', 'BuDep_3D', 'BuGir', 'BuGir_3D', 'BuDisp', 'BuDisp_3D', 'BuRan', 'BuRan_3D', 'BuRough', 'BuRough_3D', 'BuSWA_3D', 'BuSurf_3D', 'BuVol_3D', 'BuSA_3D', 'BuSWR_3D','BuEWA_3D','BuEWR_3D']
    
    if three_d:
        param_list = three_d_list
    else:
        param_list = two_d_list

    for i in param_list:
        df[[i+'_mean',i+'_median',i+'_std',i+'_min',i+'_max',i+'_sum',i+'_mode']] = momepy.describe_agg(selected_buildings[i], selected_buildings["station_id"], statistics=["mean", "median", "std", "min", "max", "sum", "mode"])
        if i != 'BuAre':    
            df[[i+'_wmean',i+'_wstd',i+'_wmedian',i+'_wmin',i+'_wmax',i+'_wsum',i+'_wper25',i+'_wper75']] = selected_buildings.groupby('station_id')[[i,weight]].apply(weighted_stats, i, weight)
        df[[i+'_IQR',i+'_MAD',i+'_skew']] = selected_buildings.groupby('station_id')[i].agg([iqr,median_abs_deviation,skew])
        df[[i+'_per25',i+'_per75']] = selected_buildings.groupby('station_id')[i].quantile([0.25,0.75]).unstack()
        df['BuNum'] = len(selected_buildings.groupby('station_id'))

    for i in ['StrLen', 'StrW', 'StrOpe', 'StrWD', 'StrH', 'StrHD', 'StrHW', 'BpM', 'StrLin', 'StrCNS']:
        df[[i+'_mean',i+'_median',i+'_std',i+'_min',i+'_max',i+'_sum' ,i+'_mode']] = momepy.describe_agg(selected_streets[i], selected_streets["station_id"], statistics=["mean", "median", "std", "min", "max", "sum", "mode"])
        df[[i+'_IQR',i+'_MAD',i+'_skew']] = selected_streets.groupby('station_id')[i].agg([iqr,median_abs_deviation,skew])
        df[[i+'_per25',i+'_per75']] = selected_streets.groupby('station_id')[i].quantile([0.25,0.75]).unstack()

    for i in ['StrClo400', 'StrBet400', 'StrMes400', 'StrGam400', 'StrCyc400', 'StrENR400', 'StrDeg', 'StrSCl']:
        df[[i+'_mean',i+'_median',i+'_std',i+'_min',i+'_max',i+'_sum' ,i+'_mode']] = momepy.describe_agg(selected_nodes[i], selected_nodes["station_id"], statistics=["mean", "median", "std", "min", "max", "sum", "mode"])
        df[[i+'_IQR',i+'_MAD',i+'_skew']] = selected_nodes.groupby('station_id')[i].agg([iqr,median_abs_deviation,skew])
        df[[i+'_per25',i+'_per75']] = selected_nodes.groupby('station_id')[i].quantile([0.25,0.75]).unstack()

    stations = stations.merge(df, left_on='station_id', right_on=df.index, how='left')
    stations['BuCAR'] = stations['BuAre_sum']/stations.geometry.area

    return stations

def mask_raster(raster_path, bldgs, new_path):
    """
    Masks a raster with building footprints

    Parameters
    ----------
    raster_path : str
        Path to the raster file
    bldgs : GeoDataFrame
        GeoDataFrame containing building footprints
    new_path : str
        Path to save the masked raster

    """
    with rasterio.open(raster_path) as src:
        crs = src.crs
        bldgs = bldgs.to_crs(crs.to_epsg())

        out_image, out_transform = mask.mask(src, bldgs.geometry, crop=False,invert=True,filled=False)

        # Define the metadata for the new file
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
            
        # Save the raster file to a different path
        with rasterio.open(new_path, 'w', **out_meta) as dest:
            dest.write(out_image)

def agg_raster(raster_path, stations, parameter_name, majority=False):
    """
    Extracts the following parameters for each station:
    - Raster mean
    - Raster median
    - Raster standard deviation
    - Raster IQR

    Parameters
    ----------
    raster_path : ndarray
        Raster path
    stations : GeoDataFrame
        GeoDataFrame containing station buffers
    parameter_name : str
        Name of the raster parameter
    majority : bool, optional
        If True, majority (mode) is calculated. The default is False.
        Requires significantly more processing time

    Returns
    -------
    stations : GeoDataFrame
        GeoDataFrame containing raster parameters

    """
    with rasterio.open(raster_path) as src:
        crs = src.crs
    stations = stations.to_crs(crs.to_epsg())

    if stations.empty:
        stations[parameter_name+'_mean'] = None
        stations[parameter_name+'_std'] = None
        stations[parameter_name+'_median'] = None
        stations[parameter_name+'_per25'] = None
        stations[parameter_name+'_per75'] = None
        stations[parameter_name+'_IQR'] = None
        stations[parameter_name+'_min'] = None
        stations[parameter_name+'_max'] = None
        stations[parameter_name+'_sum'] = None
        if majority:
            stations[parameter_name+'_majority'] = None

    else:
        if majority:
            stats = zonal_stats(stations, raster_path, stats=['mean', 'max', 'min', 'count', 'std', 'median', 'sum', 'range','percentile_25','percentile_75', 'majority'])
        else:
            stats = zonal_stats(stations, raster_path, stats=['mean', 'max', 'min', 'count', 'std', 'median', 'sum', 'range','percentile_25','percentile_75'])

        stations[parameter_name] = stats

        stations[parameter_name+'_mean'] = stations[parameter_name].apply(lambda x: x['mean'])
        stations[parameter_name+'_std'] = stations[parameter_name].apply(lambda x: x['std'])
        stations[parameter_name+'_median'] = stations[parameter_name].apply(lambda x: x['median'])
        stations[parameter_name+'_per25'] = stations[parameter_name].apply(lambda x: x['percentile_25'])
        stations[parameter_name+'_per75'] = stations[parameter_name].apply(lambda x: x['percentile_75'])
        stations[parameter_name+'_IQR'] = stations[parameter_name].apply(lambda x: x['percentile_75']) - stations[parameter_name].apply(lambda x: x['percentile_25'])
        stations[parameter_name+'_min'] = stations[parameter_name].apply(lambda x: x['min'])
        stations[parameter_name+'_max'] = stations[parameter_name].apply(lambda x: x['max'])
        stations[parameter_name+'_sum'] = stations[parameter_name].apply(lambda x: x['sum'])
        if majority:
            stations[parameter_name+'_majority'] = stations[parameter_name].apply(lambda x: x['majority'])

    return stations


def calculate_statistics(data, target_column, bootstrap = False):
    """
    Calculate Pearson and Spearman correlations, mutual information, and temperature statistics for a single time period
    Parameters
    ----------
    data : DataFrame
        DataFrame containing the data
    target_column : str
        Name of the target column
    bootstrap : bool, optional
        If True, bootstrap analysis is performed for Spearman correlation. The default is False.
    Returns
    -------
    DataFrame
        DataFrame containing the statistics for each parameter
        """

    results = []
    #print(data)
    
    # Ensure the target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # Loop through each column except the target column
    for col in data.columns:
        if col == target_column:
            continue

        #print(f"Calculating statistics for '{col}'...")

        y = data[target_column]

        std = y.std()
        mean = y.mean()
        FRKART = y[y.index == 'FRKART'].values[0]
        FRPDAS = y[y.index == 'FRPDAS'].values[0]
        FRINST = y[y.index == 'FRINST'].values[0]
        FRHBHF = y[y.index == 'FRHBHF'].values[0]
        FRHOCH = y[y.index == 'FRHOCH'].values[0] 
        FROPFS = y[y.index == 'FROPFS'].values[0]
        FRDIET = y[y.index == 'FRDIET'].values[0]
        FRTIEN = y[y.index == 'FRTIEN'].values[0]
        urban = np.mean([FRKART, FRPDAS, FRINST, FRHBHF])
        rural = np.mean([FRHOCH, FROPFS, FRDIET, FRTIEN])
        UHI_mag = urban - rural

        # Drop NA values for pairwise comparison
        valid_data = data[[col, target_column]].dropna()

        if len(valid_data) <= 2:
            # Append results
            results.append({
                'Parameter': col,
                'Pearson Correlation': None,
                'Pearson p-value': None,
                'Spearman Correlation': None,
                'Spearman p-value': None,
                'Mutual Information': None,
                'Temp. Mean': None,
                'Temp. Std. Dev.': None,
                'UHI Magnitude': None,
            })
        else:
            x = valid_data[col]
            y = valid_data[target_column]

            # Calculate Pearson correlation
            pearson_corr, pearson_pval = pearsonr(x, y)

            # Calculate Spearman's rank correlation
            spearman_corr, spearman_pval = spearmanr(x, y)
            if bootstrap:
                bs = bootstrap(spearmanr, x, y)
            else:
                bs = {}

            # Calculate mutual information
            if len(x) < 4:
                mi = None
            else:
                mi = mutual_info_regression(x.values.reshape(-1, 1), y)[0]

            # Append results
            entry = {
                'Parameter': col,
                'Pearson Correlation': pearson_corr,
                'Pearson p-value': pearson_pval,
                'Spearman Correlation': spearman_corr,
                'Spearman p-value': spearman_pval,
                'Mutual Information': mi,
                'Temp. Mean': mean,
                'Temp. Std. Dev.': std,
                'UHI Magnitude': UHI_mag}
            entry.update(bs)
            results.append(entry)

    return pd.DataFrame(results)

def stats_multiple_times(radius, vars, var, timesteps, temp):
    """
    Calculate various statistics between a variable and temperature data over multiple time periods.
    Parameters
    ----------
    radius : float
    vars : GeoDataFrame
        GeoDataFrame containing station parameters
    var : str
        Name of the variable to analyze
    time : list
        List of time period column names
    temp : DataFrame
        DataFrame containing temperature data
    Returns
    -------
        
    data : DataFrame
        DataFrame containing melted data for the variable and temperature
        """
    temp = temp.sub(temp.mean(axis=0), axis=1)
    temp = temp.div(temp.std(axis=0), axis=1)

    vars = vars.merge(temp, left_on='station_id', right_on='station_id',how='inner')
    vars["BuAdj"] = -vars["BuAdj"]  # Invert BuAdj values

    #scaler = StandardScaler()
    #vars_scaled = scaler.fit_transform(vars)
    #vars = pd.DataFrame(vars_scaled, columns=vars.columns, index=vars.index)

    data = vars[[var] + list(timesteps)].copy().reset_index()
    data = data.melt(id_vars=[var,'station_id'], value_vars=timesteps, var_name='time', value_name='temperature')
    data = data.dropna()

    if len(data) <= 2:
        spearman_corr = np.nan
        p_value = np.nan
        pearson_corr = np.nan
        r_squared = np.nan
        rmse = np.nan
        cooks_d = np.nan
        mi = [np.nan]
        y_pred = np.nan
        mean = np.nan
        std = np.nan
    
    else:
        mean = data[var].mean()
        std = data[var].std()
        
        # Compute Spearman correlation
        spearman_corr, p_value = spearmanr(data[var], data['temperature'])

        #Pearson and r squared
        pearson_corr, _ = pearsonr(data[var], data['temperature'])
        X = sm.add_constant(data[var])  # Add constant for regression
        model = sm.OLS(data['temperature'], X).fit()
        r_squared = model.rsquared

        # Get the predicted values (fitted values)
        y_pred = model.fittedvalues

        # Calculate the residuals (errors)
        residuals = data['temperature'] - y_pred

        # Calculate the least squares error (RSS)
        rss = np.sum(residuals ** 2)
        # Calculate the Mean Squared Error (MSE)
        mse = rss / len(data[var])
        # Calculate the Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)

        # Compute Cook's distance
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0].max()  # Max Cook's distance

        mi = mutual_info_regression(data[[var]], data['temperature'].values)

    return data, mean, std, spearman_corr, p_value, pearson_corr, r_squared, rmse, cooks_d, mi[0], y_pred

def _ci_index(data):
    lower = np.percentile(data, 2.5)
    upper = np.percentile(data, 97.5)

    index = (upper - lower) / (upper + lower)/2
    return index

def bootstrap(func, X, Y, n_bootstrap=1000):

    # Step 1: Calculate observed  correlation
    rho_obs, _ = func(X, Y)

    # Step 2: Bootstrap  correlation
    bootstrap_corr = []

    #reset index 
    X = X.reset_index(drop=True)
    Y = Y.reset_index(drop=True)
    
    for _ in range(n_bootstrap):
        # Resample data with replacement
        indices = np.random.choice(len(X), len(X), replace=True)
        X_bootstrap = X[indices]
        Y_bootstrap = Y[indices]
        rho_boot, _ = spearmanr(X_bootstrap, Y_bootstrap)
        bootstrap_corr.append(rho_boot)

    # Step 3: Null distribution by permuting Y
    n_null = n_bootstrap
    null_corr = []

    for _ in range(n_null):
        Y_permuted = np.random.permutation(Y)  # Shuffle Y
        rho_null, _ = spearmanr(X, Y_permuted)
        null_corr.append(rho_null)

    # Step 4: Calculate p-values
    bootstrap_corr = np.array(bootstrap_corr)
    null_corr = np.array(null_corr)

    # Bootstrap p-value
    p_bootstrap = np.mean(bootstrap_corr >= rho_obs)

    # Null p-value
    p_null = np.mean(null_corr >= rho_obs)

    # Output results
    results = {
        "Observed correlation": rho_obs,
        "Mean correlation (bootstrap)": np.mean(bootstrap_corr),
        "Standard deviation (bootstrap)": np.std(bootstrap_corr),
        "95% confidence interval (bootstrap)": np.percentile(bootstrap_corr, [2.5, 97.5]),
        "95% confidence interval index (bootstrap)": _ci_index(bootstrap_corr),
        "P-value (bootstrap)": p_bootstrap,
        "P-value (null)": p_null
    }

    return results

def plot(radius, vars, param, temp, time):
    vars = vars.merge(temp[time], left_on='station_id', right_on='station_id',how='inner')

    var = param

    plt.scatter(vars[var], vars[time])
    plt.xlabel(var)
    plt.ylabel('Temperature')
    plt.title(var+' vs Temperature')

    for i, txt in enumerate(vars.index):
        plt.annotate(txt, (vars[var][i], vars[time][i]))
    plt.show()