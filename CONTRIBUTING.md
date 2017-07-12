# How to Contribute

We welcome efforts to improve this project, and we are open to contributions for model improvements, process improvements, and general good ideas.  Please use this guide to help structure your contributions and to make it easier for us to consider your contributions and provide feedback.  If we do use your work we will acknowledge your contributions to the best of ability, and all contributions will be governed under the license terms specified in the LICENSE.md file in this project. To get started, sign the [Contributor License Agreement](https://www.clahub.com/agreements/Chicago/clear-water).

In general we use the structure provided by GitHub for our workflow management, so if you're new to GitHub please see this guide first: https://guides.github.com/activities/contributing-to-open-source/#contributing

Your contributions have the potential to have a positive impact on not just us, but everyone who is impacted by anyone who uses this project.  So, consider that a big thanks in advance.

## Reporting an Issue

Clear Water uses [GitHub Issue Tracking](https://github.com/Chicago/clear-water/issues) to track issues. This is a good place to start and can be a helpful place to manage both technical and non-technical issues. 

## Submitting Code Changes

Please send a [GitHub Pull Request to City of Chicago](https://github.com/chicago/clear-water/pull/new/master) with a clear list of what you've done (read more about [pull requests](https://help.github.com/articles/about-pull-requests/)). Always write a clear log message for your commits. 

## Demonstrating Model Performance

We welcome improvements to the analytic model that creates predictions for the Chicago Park District. The city may adopt a pull request that sufficiently improves the accuracy and prediction, thus, allowing you to contribute to the beach management
process for the City.

If your pull request is to improve the model, please consider the following steps when submitting a pull request:
* Identify how your model is improving prior results
* Run a test using the data provided in the repository
* Create a pull request which describes those improvements in the description.
* Work with the data science team to reproduce those results
 
### Training your data
Train your model using data between 2006 and 2016. If you training data is sufficiently large, use 2016 as a final validation set. Not all predictors are available for 
all years, so if your training data is not large due to your chosen predictors or preprocessing techniques, use kFolds cross-validation.

### Measuring improvement
The City seeks to increase accuracy and reduce false alarms for its water quality predictions. Thus, we are interested in a few key qualities in any improvements.
* Your model increases "hits" or true positives, where both predicted and actual bacteria levels exceeded the applicable limit.
* Your model reduces "false alarms" or false negatives, which occurs when the model predicts bacteria levels exceeded the limit, but acutal levels were below the limit.
* The City is particurly interested in a false alarm rates below 5 percent. 
* The best way to show the tradeoff between hits and false alarms is by submitting
the ROC and PR Curves.

### Ability to adopt model
If you would like to submit an improvement, please open a pull request that notes improvements to at least one of the aforementioned benchmarks. Your code should be able to reproduce those results by the data science team.

Model improvements that include new data must use data that is freely (*gratis* or *libre*) to the City of Chicago. There must not be any terms that would prohibit the City from storing data.on local servers.

Likewise, by submitting a pull request, you agree that the City of Chicago will be allowed to use your code for analytic purposes and that your software will be licensed under the licensing found in LICENSE.md in this repository.
