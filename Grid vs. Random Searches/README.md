## Grid vs. random searches

This project aims to compare two alternatives for hyper-parameters definition: grid search and random search. While **grid search** tries on all combinations of values for all hyper-parameters that should be defined during a model estimation, **random search** relies on ranges of possible values, either discrete sets of integers or intervals of real values, from which samples of randomly selected values for all hyper-parameters are sequentially tested.
<br>
<br>
Tests for the assessment involve estimation of logistic regression models and GBMs for 30 different datasets. Notebook "1 Grid vs. Random Searches - Experiments Design" presents all specifications concerning grids and distributions of values for grid search and random search, respectively. As usual, outcomes collected during tests are performance metrics (ROC-AUC, average precision score and Brier score) together with running times.
<br>
<br>
Notebook "3 Grid vs. Random Searches - Analysis of Results" discusses in-depth all results. Main findings point to some equivalence between grid and random searches regarding performance metrics. However, this same level of expected performance is reached more rapidly by random search, since this requires less estimations to cover a given density of possible values of hyper-parameters.
<br>
<br>
This folder contains Python scripts with codes for running tests alongside with Jupyter notebooks and HTML files for experiments design, methodology development and analysis of results.
