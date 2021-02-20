# Predicting avalanche risk in the Bavarian alps

## Project background

Every year, avalanches cause fatalities in mountainious regions. In Bavaria (southern Germany), the "Lawinenwarndienst Bayern" (https://www.lawinenwarndienst-bayern.de/) regularly updates an avalanche danger scale ranging from 1 (low risk) to 5 (extreme risk) to provide information about current risks. This score is established through expert considerations based on weather data of the preceding days.

Goal of this project is a) to model the avalanche danger score per region by means of aggregated weather data and b) to identify the most important variables determining avalanche risk.

## How to use this code

The Python code to model the avalanche risk scores can be downloaded from this GitHub repository. 

**Data**
Historical avalanche warning levels can be web-scraped from the Lawinenwarndienst web page by means of the code under `scrape`. In order to reproduce the modeling process, however, historical weather data have to be obtained from Lawinenwarndienst Bayern. These data cannot be shared here without permission.

**EDA**
The eda folder contains `eda_warnings.py` which provides summary statistics and plotting for the historical avalanche warning levels, and ?`eda_weather.py` which does the same thing for historical weather data. The results are also shown below.

**Preprocessing and modeling**
The folder `model` contains `model.py` which contains the code to model avalanche risk scores. Note: Because the modeling process requires tweaking many different parameter combinations, this file should be adapted and executed line by line in an IDE.

## Available data

Historical avalanche danger levels are available on https://www.lawinenwarndienst-bayern.de/res/archiv/lageberichte/. They can be web-scraped by means of the code available under `scrape`.

Historical weather data were requested from Lawinenwarndienst Bayern. This authority collects weather data specifically for understanding avalanche risk at several weather stations throughout the Bavarian alps.

## Results

### EDA

[Barplot of avalanche risk levels per zone and year](eda/output/barplot_zone_jahr.png)

## License