---
output: 
  html_document: 
    smart: no
---
# e-coli-beach-predictions
[![Stories in Ready](https://badge.waffle.io/Chicago/e-coli-beach-predictions.svg?label=ready&title=Ready)](http://waffle.io/Chicago/e-coli-beach-predictions) [![MIT License project](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)

Attempting to predict e. coli readings at Chicago beaches. The project should be reproducible, relying on scripts and avoid any manual steps.

* Repository: https://github.com/Chicago/e-coli-beach-predictions 
* Issue tracker: https://github.com/Chicago/e-coli-beach-predictions/issues 
* Project management board: https://waffle.io/Chicago/e-coli-beach-predictions 
* Documentation and notes: https://github.com/Chicago/e-coli-beach-predictions/wiki 

# Required software

This project uses [R](https://www.r-project.org/) and has the following package 
dependencies:

* dplyr
* ggplot2
* lubridate
* RSocrata
* stats
* tidyr
* randomForest

# Getting started with R

You'll need to [install R](https://cran.r-project.org/), and we recommend you 
[install RStudio](https://www.rstudio.com/products/rstudio/) as well. You can 
open this project in RStudio by opening the ```clear-water.Rproj``` file. 

If you are new to R, check out some basics [here](https://support.rstudio.com/hc/en-us/articles/201141096-Getting-Started-with-R).

# Running the model

To generate the model, open the ```Master.R``` file. Inside the file, you will
see settings that you can tweak to change the predictors and other facets of 
the model. Once you're ready, run all the code in the file. If you've successfully 
generated the model, you'll see ROC and Precision/Recall plots appear in RStudio.
You'll also have a Data Frame in R called ```model_summary``` that contains 
the results of the model evaluation.

# Running the model in production

This repo is one of two GitHub repos that make up the Clear Water project. The
other one can be found [here](https://github.com/Chicago/clear-water-app) and 
is an application the automatically generates water quality predictions based 
on daily DNA test results that are published on Chicago's Data Portal.

# Contributing

Link to CONTRIBUTING.md

# Notes

## General organization

- Data acquisition
- Data cleaning
- Functions
- Special Functions
- Master.R -> Model.R -> modelEColi.R

## A/B evaluation feature

DNA Model vs. USGS

## Chi Hack Night

- explain and link to branch

## Wiki

# LICENSE

Copyright (c) 2015 City of Chicago

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
