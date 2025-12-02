import os
import sys
import main
import geopandas as gpd
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import main


def define_lcz_colors(lcz):
    """
    Define LCZ colors and map them to station IDs.

    Parameters
    ----------
    lcz : DataFrame
        DataFrame containing station IDs and their corresponding LCZ descriptions.
    Returns
    ------- 
    dictionary
        mapping station IDs to their corresponding LCZ colors.
    """
    # Data as a list of dictionaries
    data = [
        {"LCZ": "LCZ 1", "Description": "Compact highrise", "Color": "#910613"},
        {"LCZ": "LCZ 2", "Description": "Compact midrise", "Color": "#D9081C"},
        {"LCZ": "LCZ 3", "Description": "Compact lowrise", "Color": "#FF0A22"},
        {"LCZ": "LCZ 4", "Description": "Open highrise", "Color": "#C54F1E"},
        {"LCZ": "LCZ 5", "Description": "Open midrise", "Color": "#FF6628"},
        {"LCZ": "LCZ 6", "Description": "Open lowrise", "Color": "#FF985E"},
        {"LCZ": "LCZ 7", "Description": "Lightweight low-rise", "Color": "#FDED3F"},
        {"LCZ": "LCZ 8", "Description": "Large lowrise", "Color": "#BBBBBB"},
        {"LCZ": "LCZ 9", "Description": "Sparsely built", "Color": "#FFCBAB"},
        {"LCZ": "LCZ 10", "Description": "Heavy Industry", "Color": "#565656"},
        {"LCZ": "LCZ 11 (A)", "Description": "Dense trees", "Color": "#006A18"},
        {"LCZ": "LCZ 12 (B)", "Description": "Scattered trees", "Color": "#00A926"},
        {"LCZ": "LCZ 13 (C)", "Description": "Bush, scrub", "Color": "#628432"},
        {"LCZ": "LCZ 14 (D)", "Description": "Low plants", "Color": "#B5DA7F"},
        {"LCZ": "LCZ 15 (E)", "Description": "Bare rock or paved", "Color": "#000000"},
        {"LCZ": "LCZ 16 (F)", "Description": "Bare soil or sand", "Color": "#FCF7B1"},
        {"LCZ": "LCZ 17 (G)", "Description": "Water", "Color": "#656BFA"}
        ]
    # Create DataFrame
    df = pd.DataFrame(data)
    df['LCZ_number'] = df['LCZ'].str.extract(r'(\d+)').astype(int)

    lcz = lcz[['station_id','local_climate_zone']]
    # extract description in brackets
    lcz['LCZ_description'] = lcz['local_climate_zone'].str.extract(r'\((.*?)\)')
    # merge with df on LCZ_description to get color and number
    lcz = lcz.merge(df, left_on='LCZ_description', right_on='Description', how='inner')

    # make a dictionary out of station_id and color
    lcz_colors_dict = dict(zip(lcz['station_id'], lcz['Color']))
    return lcz_colors_dict

def custom_lcz_legend():
        """
        Create a custom legend for LCZ colors.
        
        Returns
        -------
        list
            List of Line2D objects for custom legend.
        """
        lcz_colors = {
             
        "LCZ 1: Compact highrise": "#910613",
        "LCZ 2: Compact midrise": "#D9081C",
        "LCZ 3: Compact lowrise": "#FF0A22",
        "LCZ 4: Open highrise": "#C54F1E",
        "LCZ 5: Open midrise": "#FF6628",
        "LCZ 6: Open lowrise": "#FF985E",
        "LCZ 7: Lightweight low-rise": "#FDED3F",
        "LCZ 8: Large lowrise": "#BBBBBB",
        "LCZ 9: Sparsely built": "#FFCBAB",
        "LCZ 10: Heavy Industry": "#565656",
        "LCZ 11 (A): Dense trees": "#006A18",
        "LCZ 12 (B): Scattered trees": "#00A926",
        "LCZ 13 (C): Bush, scrub": "#628432",
        "LCZ 14 (D): Low plants": "#B5DA7F",
        "LCZ 15 (E): Bare rock or paved": "#000000",
        "LCZ 16 (F): Bare soil or sand": "#FCF7B1",
        "LCZ 17 (G): Water": "#656BFA"
        }

        lcz_colors = {
             
        "LCZ 2: Compact midrise": "#D9081C",
        "LCZ 4: Open highrise": "#C54F1E",
        "LCZ 5: Open midrise": "#FF6628",
        "LCZ 6: Open lowrise": "#FF985E",
        "LCZ 8: Large lowrise": "#BBBBBB",
        "LCZ 9: Sparsely built": "#FFCBAB",
        "LCZ 11 (A): Dense trees": "#006A18",
        "LCZ 12 (B): Scattered trees": "#00A926",
        "LCZ 14 (D): Low plants": "#B5DA7F",
        "LCZ 17 (G): Water": "#656BFA"}

        lcz_labels = list(lcz_colors.keys())
        lcz_colors = list(lcz_colors.values())
        
        # Create custom legend
        custom_lines = [plt.Line2D([0], [0], marker='o', color='w', label=lcz_labels[i],
                                        markerfacecolor=lcz_colors[i], markersize=10) for i in range(len(lcz_labels))]
        return custom_lines

def simple_plot_reduced(ax, radius, station_params, var, time, temp, stations, var_name_mapping):
    """
    Create a scatter plot of a variable against temperature with statistical annotations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    radius : int
        The radius for data selection.
    station_params : GeoDataFrame
        GeoDataFrame containing station parameters.
    var : str
        The variable to plot against temperature.
    time : list
        List of time period column names
    temp : DataFrame
        DataFrame containing temperature data
    """

    data, mean, std, spearman_corr, p_value, pearson_corr, r_squared, rmse, cooks_d, mi, y_pred = main.stats_multiple_times(radius, station_params, var, time, temp)

    print(f"Spearman ρ: {spearman_corr:.2f}\nMutual Info.: {mi:.2f}")

    # Add textbox with correlation and Cook’s distance
    textstr = (
        fr'$\rho_{{fix,300,\langle UHI\rangle}} = {spearman_corr:.2f}$' '\n'
        fr'$\mathrm{{MI}}_{{fix,300,\langle UHI\rangle}} = {mi:.2f}$'
    )
    ax.text(0.6, 0.05, textstr, transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='grey', facecolor='none'))

    lcz_colors = define_lcz_colors(stations)
    print(lcz_colors)
    colors = [lcz_colors[station] for station in data['station_id']]  # Assign colors to each station
    ax.scatter(data[var], data['temperature'], marker ='x', c=colors, alpha =0.5, label = data['station_id'])
    #ax.plot(data[var], y_pred, color='black', linewidth=1)  # Plot regression

    ax.set_xlabel(var_name_mapping[var],fontsize=16)
    ax.set_ylabel('Standardised Temperature',fontsize=16)
    #ax.set_title(var+' vs Temperature'+' for '+str(radius)+'m radius')

    for i, txt in enumerate(data['station_id'].unique()):
        ax.annotate(txt, (data[data['station_id'] == txt][var].iloc[0], data[data['station_id'] == txt]['temperature'].iloc[0]), color=lcz_colors[txt])

