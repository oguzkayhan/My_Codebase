# My_Codebase
This repository contains a collection of useful functions, classes, and code blocks for data science projects in Python. 
The contents of this repository are organized into the following directories:

### FeatureSelector: 
  This directory is a tool for selecting important features from a dataset using a given model and a specified importance threshold.

### OptimalCorrCleaner:
  This function helps to remove highly correlated features with low importance from a DataFrame. It considers feature importance, finds high correlation pairs, and removes the less important feature from the pair. If a feature has more than one high correlation pair and it is dropped due to low importance, the function skips the other pairs. In this way, undesired feature elimination is prevented. The function returns a list of remaining features and a list of dropped features.
### utils: 
  This directory contains miscellaneous utility functions and classes.
