# Quick Start

You need to have [numpy](http://www.numpy.org/), [pandas](http://pandas.pydata.org/), [scikit-learn](http://scikit-learn.org/stable/), and [matplotlib](http://matplotlib.org/) installed.

If all of these dependencies are installed, then you can run `python modeling.py -v` and you will see some values being printed to the terminal (the `-v` is the verbose flag). Eventually, you should also see two plots, a [ROC (Receiver Operator Characteristic) curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) and a [PR (Precision Recall) curve](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html). You can tweak the model inputs by changing parts of `modeling.py`, the `prepare_data` function is a good place to start looking, a snippet is included below, note that it is optimized for easily commenting out individual variables.

```python
# Deterministic columns are known ahead of time, their actual values can be used.
deterministic_columns = [
    # 'Client.ID',  # subsumed by the geographic flags

    'precipIntensity',
    'precipIntensityMax',
    'temperatureMin',
    'temperatureMax',
    'humidity',
    'windSpeed',
    'cloudCover',

    # 'sunriseTime',  # commenting for now since it is in absolute UNIX time

    # 'Days.Since.Last.Holiday',

    'flag_geographically_a_north_beach',
    'categorical_beach_grouping'
]

# Deterministic columns are known ahead of time, their actual values are used.
# These hourly variables have an additional parameter which defines what hours
# should be used. For example, an entry
#   'temperature':[-16,-13,-12,-11,-9,-3,0]
# would indicate that the hourly temperature at offsets of
# [-16,-13,-12,-11,-9,-3,0] from MIDNIGHT the day of should be included as
# variables in the model.
deterministic_hourly_columns = {
    'temperature':range(-19,5),
    'windSpeed':[1,2,3,4],
    'windBearing':[4],
    'pressure':[0],
    'cloudCover':[4],
    'humidity':[4],
    'precipIntensity':[-14,-13,-12,-11,-10,0,4]
}
for var in deterministic_hourly_columns:
    for hr in deterministic_hourly_columns[var]:
        deterministic_columns.append(var + '_hour_' + str(hr))

# Historical columns have their previous days' values added to the predictors,
# but not the current day's value(s) unless the historical column also exists
# in the deterministic columns list.
# Similar to the hourly columns, you need to specify which previous days
# to include as variables. For example, below we have an entry
#   'temperatureMax': range(1,4)
# which indicates that the max temperature from 1, 2, and 3 days previous
# should be included.
historical_columns = {
    'temperatureMin': range(1,3),
    'temperatureMax': range(1,4),
    # 'humidity': range(1,3),
    # 'windSpeed': range(1,8),
    # 'cloudCover': range(1,8),
    'Escherichia.coli': range(1,8)
}
historical_columns_list = list(historical_columns.keys())
```




You can also change the hyperparameters of the model (and the classifier, which currently defaults to `sklearn.ensemble.RandomForestClassifier`) in the `if __name__ == '__main__':` block of `modeling.py`:

```python
if __name__ == '__main__':

    ...

    # Set up model parameters
    hyperparams = {
        # Parameters that effect computation
        'n_estimators':2000, # even with 2000, still moderate variance between runs
        'max_depth':6,
        'class_weight': {0: 1.0, 1: 1/.15},
        # Misc parameters
        'n_jobs':-1,
        'verbose':False
    }
    partial_auc_bounds = [0.005, 0.05]
    clfs, auc_rocs, roc_ax, pr_ax = model(timestamps, predictors, classes,
                                          classifier=sklearn.ensemble.RandomForestClassifier,
                                          hyperparams=hyperparams,
                                          roc_bounds=partial_auc_bounds,
                                          verbose=args.verbose)
```

If you are making small tweaks to the model, and wish to avoid loading and cleaning the data during each run, then you can load and clean the data once, storing the output in `tmp.csv` by running `python read_data.py -o tmp.csv`. (Be aware that this will create a 50 MB CSV file.) You can then run the modeling code with `python modeling.py -i tmp.csv -v` and it will load `tmp.csv` directly.

# Details

## read_data.py

This file reads in data from all the different sources (excel spreadsheets/CSV files, over the internet, etc.) and has some limited utilities for creating derived variables such as date-shifting columns (so that you can include yesterday's weather as a predictor, for example). It is primarly meant to be used as a python module, but has a rudimentary Command Line Interface (CLI). The usage is copied below.

```
usage: read_data.py [-h] [-o outfile] [-v] [-t]

Process beach data.

optional arguments:
  -h, --help            show this help message and exit
  -o outfile, --outfile outfile
                        output CSV filename
  -v, --verbose
  -t, --test            run tests
```

Of note, `python read_data.py -o outfile` will save a CSV with the given name. Running `python read_data.py -t` will run simple unit tests (which only check to make sure everything can run to completion, and does not check for accuracy!).


## modeling.py

This file runs the classification implementing a Leave-One-Year-Out-Cross-Validation performance check. This file too has a rudimentary CLI:

```
usage: modeling.py [-h] [-i input_filename] [-v]

Process beach data.

optional arguments:
  -h, --help            show this help message and exit
  -i input_filename, --input input_filename
                        input CSV filename
  -v, --verbose
```

If you want to change the classifier, then you will have to edit the source code directly. You should not need to change the function `model`, but you might want to change the function `prepare_data` which controls what columns are used as predictors. There are also hyperparameters to change in the `if __name__=='__main__':` block. See the Quick Start section above.


## visualizations.py

This module includes a lot of plotting utilities useful for judging model performance or data exploration.

## data_investigations.py

This module includes code to check the time of day each beach was sampled.
