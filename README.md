# Troposcatter trace calculation

## Prerequirements

You need to get an https://rapidapi.com/ API key for using [elevations API](https://rapidapi.com/toursprung-toursprung-default/api/maptoolkit/playground/apiendpoint_6da0665d-de84-4227-a41d-accd9c7974d7) and API key for using [declinations API](https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#declination) 

## Using
Install dependencies
```sh
poetry install
```
Activate virtual environment
```sh
poetry shell
```
Run tests (verbose)
```sh
pytest -v -s
```
Run script
```sh
python trace_calc/main.py
```
