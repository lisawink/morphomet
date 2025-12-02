# morphomet
## MorphoMet - Linking Urban Morphometrics and Meteorology

## Introduction

MorphoMet is a library for quantitive analysis of the relationship between urban form and observational urban climate data. It calculates urban form parameters from building footprints and street network (using libraries like [GeoPandas](http://geopandas.org) and [momepy](https://docs.momepy.org/en/stable/)) and aggregates them to circular buffers around weather station locations (fixed or mobile). It is written for air temperature data but can be applied to other variables such as relative humidity.

## Analyses

Spearman correlation between morphometric variables and a weather variable for multiple locations in a city can be calculated for a single measurement time or for multiple timesteps. Methodology details are given in our [paper].

## Notebooks
- For preprocessing data, [0_preprocess_data.ipynb](notebooks/0_preprocess_data.ipynb)
- For caculating parameters, [1_calc_params.ipynb](notebooks/1_calc_params.ipynb)
- For calculating fractions of night and synoptically-driven conditions, [2_synoptic_control.ipynb](notebooks/2_synoptic_control.ipynb)
- For correlations,
- For figures from the paper, 

## Functions
- Main functions are given in [main](src/main.py)
- Functions used for visualisations are given in [vis](src/vis.py)