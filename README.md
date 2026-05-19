# MorphoMet 

## Linking Urban Morphometrics and Meteorology

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20283958.svg)](https://doi.org/10.5281/zenodo.20283958)

## Quick Start

```bash
git clone https://github.com/lisawink/morphomet.git
cd morphomet
conda env create -f environment.yml
conda activate morphomet
jupyter notebook
```

## Introduction

MorphoMet is a research workflow for analysing the relationship between urban form and observational urban climate data. It calculates urban form parameters from building footprints and street networks (using libraries like [GeoPandas](http://geopandas.org) and [momepy](https://docs.momepy.org/en/stable/)) and aggregates them to circular buffers around weather station locations (fixed or mobile). The workflow is designed for air temperature data but can be applied to other variables such as relative humidity.

## Analyses

MorphoMet supports statistical analysis of relationships between urban morphometric variables and meteorological observations across multiple locations and timesteps. Current workflows include Spearman correlation analyses for individual measurement times as well as temporally resolved analyses across full measurement periods.

Methodological details are provided in our [paper].

## Notebooks

- For preprocessing data, [0_preprocess_data.ipynb](notebooks/0_preprocess_data.ipynb)
- For calculating parameters, [1_calc_params.ipynb](notebooks/1_calc_params.ipynb)
- For calculating fractions of night and synoptically-driven conditions, [2_synoptic_control.ipynb](notebooks/2_synoptic_control.ipynb)
- For statistics per timestep, [3_model.ipynb](notebooks/3_model.ipynb)
- For figures from the paper, [notebooks/visualisations/](notebooks/visualisations/)

## Functions

- Main functions are given in [main](src/main.py)
- Functions used for visualisations are given in [vis](src/vis.py)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/lisawink/morphomet.git
cd morphomet
```

### 2. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate morphomet
```

### 3. Launch jupyter

```bash
jupyter notebook
```
Some notebooks require packages that are not available through conda and must be installed separately using `pip`:
```bash
pip install neatnet
pip install geoplanar
```

## Citation

If you use MorphoMet in your research, please cite the associated Zenodo release:

Winkler, L. (2026). MorphoMet (v0.1.1) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20283958