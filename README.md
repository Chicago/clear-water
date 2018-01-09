# Clear Water
[![DOI](https://zenodo.org/badge/41771713.svg)](https://zenodo.org/badge/latestdoi/41771713) [![MIT License project](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)


The City of Chicago's Clear Water project brings an innovative approach to beach water quality monitoring. It uses a machine learning prediction technique to better forecast the bacteria levels at Chicago beaches. The model works by interpreting patterns in the results of DNA tests at a handful of beaches across the City, which are then extrapolated to forecast the water quality at other, untested beaches. This method provides a new way for beach managers to save money on expensive rapid water quality tests.

Initial evaluation of the model has shown a significant improvement over current methods of predicting beach water quality. Testing is ongoing, and the 2017 beach season is being analyzed to further improve and evaluate the model's performance.

## Getting started with R

This project uses [R](https://www.r-project.org/), which you can [download here](https://cran.r-project.org/). We recommend you also
[install RStudio](https://www.rstudio.com/products/rstudio/). You can 
open this project in RStudio by opening the ```clear-water.Rproj``` file. 

If you are new to R, check out some basics [here](https://support.rstudio.com/hc/en-us/articles/201141096-Getting-Started-with-R).

## Running the model

To generate the model, open the ```Master.R``` file. Inside the file, you will
see settings that you can tweak to change the predictors and other facets of 
the model. Once you're ready, run all the code in the file. If you've successfully 
generated the model, you'll see ROC and Precision/Recall plots appear in RStudio.
You'll also have a Data Frame in R called ```model_summary``` that contains 
the results of the model evaluation.

## Running the model in production

This repo is one of two GitHub repos that make up the Clear Water project. The
other one can be found [here](https://github.com/Chicago/clear-water-app) and 
is an application that automatically generates water quality predictions based 
on daily DNA test results that are published on Chicago's Data Portal.

## Contributing

If you are interested in contributing to this project, see our [Contribution Guide](https://github.com/Chicago/clear-water/blob/master/CONTRIBUTING.md).

## Notes

### Collaboration with the Civic Tech Community

This project originated as a [breakout group](https://chihacknight.org/projects/2016/05/01/e-coli-predictions.html) at [Chi Hack Night](https://chihacknight.org/about.html).

### Resources

* Repository: https://github.com/Chicago/clear-water
* Issue tracker: https://github.com/Chicago/clear-water/issues 
* Project management board: https://waffle.io/Chicago/clear-water 
* Documentation and notes: https://github.com/Chicago/clear-water/wiki 

### LICENSE

Copyright (c) 2015 City of Chicago

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
