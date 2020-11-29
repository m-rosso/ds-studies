
## Tests over GBM hyper-parameters

This notebook presents all results and discussions that follow tests conducted in order to study the specification of GBM hyper-parameters.
<br>
In a first place, theoretical aspects of GBM method and data modeling are discussed in sections "Gradient Boosting Model (GBM) and its main hyper-parameters" and "Theoretical framework". Then, the objectives and structure of tests are presented, after which follow the main conclusions from the tests. The remaining of the notebook has full results.
<br>
From these tests over GBM hyper-parameters, two additional studies have emerged: the first compares values for the performance metrics of precision-recall AUC and average precision score (folder older "Peformance metrics for classification tasks"), while the second opposes grid searches using ROC-AUC and average precision score (folder "Grid Search Metrics"). Full results of both such studies are presented and discussed in their own notebooks, even though an appendix here shows preliminary results for the second study, making use of the same outcomes from tests over GBM hyper-parameters.

-----------

#### Gradient Boosting Model (GBM) and its main hyper-parameters

In their standard setting, **Gradient Boosting Models (GBMs)** are tree-based learning algorithms that produce an ensemble of different estimators in order to provide more robust predictions. Analogous to bagging and random forests, predictions from GBM are constructed upon a collection of individual estimators, but differently from them, learners that compose the GBM ensemble are not independent from each other, since they are defined in a sequential and evolutional manner.
<br>
<br>
Consequently, it follows that GBM estimation requires: i) creating $M$ different (but not independent) estimators; ii) combining them to conceive a final model. A first question that emerges from such procedure is *what data to use at each estimation* $m$? All $N$ training data points available, or just some fraction $\eta$ of randomly picked observations? This hyper-parameter $\eta$ is named **subsample** and it is defined in the interval $(0, 1]$.
<br>
<br>
Once defined $\eta = 1$ or $\eta < 1$, and in the second case which specific value $\eta \in (0,1)$ to use, given that each base learner that constitute the GBM ensemble is a decision-tree, a second choice to do concerns the size of each tree, which can be understood in different ways: i) number of terminal nodes; ii) number of splits. The first definition is how Friedman, Hastie, and Tibshirani (2008) deal with tree sizes, and it actually is more intuitive to grasp a tree size. Number of splits, in its turn, can be translated into the depth of a tree. Either ways of defining a tree size are highly related, but the number of splits more directly reveals the possible degree of interaction between input variables. Thus, another relevant hyper-parameter for GBM is the **maximum depth** of trees, $max\_depth \in \mathbb{N}_+$, varying from $max\_depth = 1$, where trees are actually stumps and there is no interaction among inputs of a given tree, and $max\_depth > 1$, where interaction effects may be captured by single trees.
<br>
<br>
Since GBM is a kind of ensemble model, the contribution of each base learner to the final composite model can also be calibrated. The hyper-parameter that controls this is the **learning rate**, or shrinkage parameter, $v \in (0, 1]$. Defining $v < 1$ leads to a regularized model, and $v \rightarrow 0$ implies in a slow learning that attenuates the contribution of each tree in the ensemble, thus preventing overfitting caused by specificities of data expressed in a few, but eventually influent base learners.
<br>
<br>
Finally, being defined how data is used in each estimation, which kind of trees can be estimated, and the weight each of them receives when composing the final model, it is necessarily to declare how many of such individual estimators should be constructed. This last hyper-parameter is the **number of estimators**, $n\_estimators \in \mathbb{N}_+$, where its value is typically large, $n\_estimators \geq 100$.
<br>
<br>
**Note:** besides these hyper-parameters specific for ensemble models, there are several others concerning trees construction.

Therefore, the main hyper-parameters to be explored when estimating GBM are:
* Subsample, $\eta$: whether $\eta = 1$ or $\eta < 1$, and which value when $\eta \in (0, 1)$.
* Maximum depth, $max\_depth$: choosing among $\{1, 2, ..., 10\}$.
* Learning rate, $v$: exploring different values smaller than 1.
* Number of estimators, $n\_estimators$.

---------------

#### Theoretical framework

Prior to presenting objectives and structure of tests, it is necessary to define some crucial objects concerning model estimation. First of all, a **statistical model** consists on a function $F(X)$ that defines how a response variable $Y$ is defined from inputs $X$ through **parameters** $\gamma$, which ultimately defines a model. To this deterministic relationship, an irreducible random error $\epsilon$ is conceived. How these parameters $\gamma$ relate with inputs $X$ depends on the **statistical learning method** used to estimate such parameters from empirical data. Estimation methods of any complexity levels are reference by **hyper-parameters** $\theta$, which are not estimated from data, but rather defined previously to estimation. So, before estimating a model, one should collect data, choose which method to use and define which values its hyper-parameters will assume.
<br>
<br>
Given a statistical learning method $L$ based on hyper-parameters $\theta^L \in \Theta^L$, where $\Theta^L \subset \mathbb{R}^k$, and parameters $\gamma^L \in \mathbb{R}^p$, the **model space** $\mathcal{M}^L$ is understood to be a p-dimensional Euclidean space accomplishing all possible values for parameters $\gamma^L$. When some algorithm is about to execute the estimation of a model under the specific statistical learning framework $L$, the model probability distribution $P(\gamma^L|\theta^L)$ is a direct function of hyper-parameters $\theta^L$ that must be defined previously to the estimation of $\gamma^L$.
<br>
This means that how likely it is to some model $\hat{\gamma}^L \in \mathcal{M}^L$ to be estimated depends on the hyper-parameters choice $\theta^L$ - and, of course, on the data being modeled. Note that, depending on the statistical learning method under consideration, such model probability distribution may be constant, i.e., no randomness would exist in model estimation. Even so, its hyper-parameters will define which given model $\hat{\gamma}$ will be estimated (given the data), but in a deterministic way.
<br>
Note also that the objective, in supervised learning tasks, is to approximate the target function $F(X)$ that defines precisely how the response variable $Y$ relates with inputs $X$. Therefore, the main concern is to define the best learning method and, then, the best hyper-parameter vector $\theta^{L*} \in \Theta^L$ that would lead to an expected estimated model $E(\hat{\gamma}^L|\theta^{L*})$ that gets the closest to $F(X)$.
<br>
Consequently, finding such $\theta^{L*}$ requires trying out all possible combinations of $\theta_1$, $\theta_2$, ..., $\theta_k$ available in $\Theta^L$. In this sense, to say that some value of $\theta_j$ is the "best", or "appropriate" is stricly correct only if all others $\theta_{-j}$ are also defined in their best values. Such *general perspective*, however, it is not only unfeasible, but, which is more important here, is also uninformative.
<br>
Therefore, tests as those whose results are presented and discussed here assume a *partial perspective*, since they assume all other things being equal. If their results do not necessarily reveal hyper-parameters values that guarantee the best possible expected performance, they are still able to reference good strategies for hyper-parameters specification, besides of providing evidence of how predictive accuracy should relate with main hyper-parameters of a given statistical learning method.

------------

#### Objetives and structure of tests

Turning the attention back to tests over GBM hyper-parameters, the objectives of the study whose results are presented here are as follows:
1. Oppose theoretical considerations to empirical evidence.
    * For instance, Friedman, Hastie, and Tibshirani (2008) indicate that hardly large trees would lead to better results than shorter ones (page 363 of 2nd edition). They also point to the relevance of defining a very small learning rate (page 365) and to the possibility of stochastic GBM to outperform standard GBM (page 365) - thus, to the possibility of $subsample < 1$ be preferable to $subsample = 1$.
<br>
<br>
2. Explore different ranges of values for those main hyper-parameters, so that appropriate values for each of them can be assessed - again, under a partial perspective. Having such optimal values at hand, one can either perform:
    1. Grid or random search over a pre-selected set of values for a given hyper-parameter $\theta_j$.
    2. Grid or random search over a set of values for hyper-parameters $\theta_{-j}$, while $\theta_j$ is fixed in some appropriate value.

Concerning struture of tests, algorithms from *sklearn* library were used, while data pre-processing, transformations and validation procedures followed codes autonomously derived. The response variable was binary $Y \in \{0,1\}$ for each one of the 100 different datasets, all of which are more-or-less unbalanced and presented different sets of input variables.
<br>
The setting was the following for each hyper-parameter explored:
1. **Subsample:**
    * $\eta \in \{0.75, 0.8, 0.9, 1\}$.
    * $max\_depth = 3$.
    * $learning\_rate = 0.05$.
    * $n\_estimators = 500$.
<br>
<br>
2. **Max depth:**
    * $\eta = 1$.
    * $max\_depth \in \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$.
    * $learning\_rate = 0.1$.
    * $n\_estimators \in \{500, 1000\}$.
<br>
<br>
3. **Learning rate:**
    * First setting: small values:
        * $\eta = 1$.
        * $max\_depth = 3$.
        * $learning\_rate \in \{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1\}$.
        * $n\_estimators = 500$.
    * Second setting: moderate values:
        * $\eta = 1$.
        * $max\_depth = 3$.
        * $learning\_rate \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$.
        * $n\_estimators \in \{100, 250, 500\}$.
<br>
<br>
4. **Number of estimators:**
    * $\eta = 1$.
    * $max\_depth = 3$.
    * $learning\_rate \in \{0.05, 0.1\}$.
    * $n\_estimators \in \{100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000\}$.

This setting was based on some default values from sklearn library, such as $max\_depth = 3$ or $learning\_rate = 0.1$, except when these hyper-parameters were subject to grid search. Given the trade-off existing between learning rate and number of estimators, when exploring learning rate, different number of estimators were used for moderate values ($0.1 \leq learning\_rate \leq 0.9$). For the same reason, when exploring number of estimators, two different learning rates were used.
<br>
<br>
When it comes to methodology for outcomes assessment, both descriptive statistics and data visualization were applied, having as performance metric of reference the ROC-AUC statistic, although precision average score, precision-recall AUC, Brier score, and binomial deviance were also analyzed.
<br>
<br>
A final remark must discuss generalization of the results presented here. Any conclusion should be taken with proper caution, since datasets used were limited with respect to their nature. Tests with a similar setting applied to more diverse data - different binary response variable, multiclass classification, regression problem - and even to simulated datasets would largely contribute to the robustness of results.
<br>
<br>
Irrespective of datasets nature, some additional procedures have the ability to improve the tests. When studying hyper-parameter $\theta_j^L$, instead of defining $\theta_{-j}^L$ to ad-hoc values, as done here, it is possible to previously define $\theta_{-j}^L$ through grid or random search using cross-validation on training data and some reference value for $\theta_j^L$. Later, train-test split validation would lead to performance metrics for a grid of $\theta_j^L$ values, using the best values for $\theta_{-j}^L$ obtained during cross-validation estimation. Besides, in order to reduce variance of results, it would be beneficial to only consider datasets with sufficiently large number of observations, or then to previously select features for small length datasets, in order to attenuate dimensionality problems.

-----------------
<a id='main_conclusions'></a>

#### Main conclusions

In this section, main conclusions derived from analysis of results are presented and discussed.
1. **Subsample:** the smaller the trainig data, the smaller it is its faithfulness with respect to the population it represents. Thus, impose more randomness to the set of observations used when producing each estimator in the ensemble may reduce the variability component of predictive performance, even more than compensating the increasing in bias. Furthermore, using subsample with big datasets reduces running time, while preserving a faithful sample of data. As a result, $\eta < 1$ is an appropriate choice when defining GBM.
    * Best average performance metrics for $\eta < 1$. In particular, 0.8, 0.9 and 0.75 presented the highest averages of test ROC-AUC, respectively, with quite similar values.
        * [Reference 1](#reference1)<a href='#reference1'></a>: averages of performance metrics by hyper-parameter value.
        * [Reference 2](#reference2)<a href='#reference2'></a>: boxplots of performance metric by subsample value.
    * Since it seems best to define a subsample value smaller than 1, these possibilities distribute among different values, which explains why $\eta = 1$ has the largest share of best performances. Even so, $\eta = 0.9$ and $\eta = 0.75$ have a very close share.
        * [Reference 3](#reference3)<a href='#reference3'></a>: count plot of best hyper-parameter value.
    * $\eta = 1$ is prone to be the best hyper-parameter value on datasets for which there is a natural tendency for a good classification performance.
        * [Reference 4](#reference4)<a href='#reference4'></a>: stripplot of performance metric by best hyper-parameter value.
    * Consequently, $\eta < 1$ seems specially promising for datasets whose classification task is more challenging, as those with few observations, more unbalanced datasets, etc.
        * [Reference 5](#reference5)<a href='#reference5'></a>: average test ROC-AUC by best subsample value (note: this reference does not point to any causality between subsample and performance metric).
        * [Reference 6](#reference6)<a href='#reference6'></a>: averages number of observations and response variable by best hyper-parameter value.
        * [Reference 7](#reference7)<a href='#reference7'></a>: count plot of best subsample by quartiles of number of observations.
        * [Reference 8](#reference8)<a href='#reference8'></a>: heatmap of correlation matrices for different subsamples.
    * Even though not individually concerning subsample hyper-parameter, it was found a positive and concave relationship between dataset length and predictive performance.
        * [Reference 9](#performance_data_info)<a href='#performance_data_info'></a>: scatterplot of performance metric against number of observations.
<br>
<br>
2. **Maximum depth:** high values for maximum depth increase model complexity, and depending on the dataset, this enlarged complexity may lead to overfitting due to the capture of interaction effects only present on training data. Consequently, tree size is a hyper-parameter whose range of values is expected to produce very distinct performance metrics. A range of $\{1, 2, 3, 4, 5\}$ seems reasonable to be explored through grid-search in most applications.
    * Clearly, performance was distinctly better for small values of $max\_depth$.
        * References [10](#reference10)<a href='#reference10'></a> and [11](#reference11)<a href='#reference11'></a>: averages of performance metrics.
    * $max\_depth \in \{1, 2\}$ concentrate more than a half of best hyper-parameter values.
        * [Reference 12](#reference12)<a href='#reference12'></a>: count plot of best hyper-parameter value.
    * Datasets whose best $max\_depth$ is high are likely to have better average performance.
        * [Reference 13](#reference13)<a href='#reference13'></a>: stripplot of performance metric by best hyper-parameter value.
    * Small $max\_depth$ values have particularly good performance with small datasets.
        * [Reference 14](#reference12)<a href='#reference12'></a>: count plot of best hyper-parameter value by quartiles of number of observations.
    * Smaller correlation between performance metric and number of observations for small values of $max\_depth$.
        * [Reference 15](#reference15)<a href='#reference15'></a>: heatmap of correlation matrices for different maximum depth values.
<br>
<br>
3. **Learning rate:** smaller learning rates are expected to produce better results, even if they require a relatively high number of estimators in order to properly explore the highest possible quantity of patterns that exist on training data. Some good options to be explored are $\{0.01, 0.05, 0.1\}$, depending on data length.
    * Indeed, $v \leq 0.1$ has substantially higher performance metrics than $v > 0.1$, irrespective of the number of estimators used.16
        * [Reference 16](#reference16)<a href='#reference16'></a>: averages of performance metric by learning rate.
    * Learning rates $v \leq 0.1$ have similar frequencies of best hyper-parameter value. Even so, $v = 0.01$ - the smallest value tested - presents a distinguished performance, given its distribution of performance metrics and its share of best hyper-parameter value.
        * References [17](#reference17)<a href='#reference17'></a> and [18](#reference18)<a href='#reference18'></a>: count plot of best hyper-parameter value and boxplot of performance metric by learning rate.
    * Decreasing correlation between performance metric and number of observations, and increasing with number of features across learning rate values.
        * [Reference 19](#reference19)<a href='#reference19'></a>: heatmap of correlation matrices for different maximum depth values.
<br>
<br>
4. **Number of estimators:** this hyper-parameter should accommodate sufficiently small learning rates. In general, and similarly to the number of training epochs on neural network modeling, it seems reasonable to adjust the number of estimators so that predictive performance can be optimized, given good choices for the remaining hyper-parameters. In general applications, no more than 500 estimators appears to be sufficient to produce good estimations.
    * Very similar results for a broad range of values.
        * [Reference 20](#reference20)<a href='#reference20'></a>: boxplots of performance metric.
    * Moderate values for the number of estimators prevail as optimal values across analyzed datasets (e.g., $M \in \{100, 250\}$. Besides, large values presented a small share of best hyper-parameters.
        * [Reference 21](#reference21)<a href='#reference21'></a>: count plot of best hyper-parameters.
    * Relevance of early stopping as a validation procedure when estimating GBM: datasets whose best hyper-parameter value is large show more potential to have a good predictive performance.
        * [Reference 22](#reference22)<a href='#reference22'></a>: stripplot of performance metric against number of estimators.
    * Relevance of early stopping as a validation preocedure when estimating GBM: small datasets are more likely to select small values of number of estimators, as compared to larger datasets - however, even for these datasets, still prevail moderate values.
        * [Reference 23](#reference21)<a href='#reference21'></a>: count plot of best hyper-parameters.

Having these conclusions in mind, two immediate possibilities emerge: i) to use the appropriate values found in applications where the predictive accuracy need not to be the best possible, but at least reasonably good accuracies are needed in order to compare performances of reference against those acquired by modifying data modeling in any way; ii) to compare the quality of results obtained using the appropriate values found taken as benchmark the performances derived from random search (study to be implemented in the future).

---------------

**Summary:**
1. [Libraries](#libraries)<a href='#libraries'></a>.
2. [Functions](#functions)<a href='#functions'></a>.
3. [Importing data](#imports)<a href='#imports'></a>.
4. [Subsample](#subsample)<a href='#subsample'></a>.
    * [Processing data](#proc_data_subsample)<a href='#proc_data_subsample'></a>.
    * [Statistics by hyper-parameter value](#stats_subsample)<a href='#stats_subsample'></a>.
    * [Describing hyper-parameter values](#describing_subsample_values)<a href='#describing_subsample_values'></a>.
    * [Data visualization](#data_vis_subsample)<a href='#data_vis_subsample'></a>.
<br>
<br>
5. [Max depth](#max_depth)<a href='#max_depth'></a>.
    * [Processing data](#proc_data_max_depth)<a href='#proc_data_max_depth'></a>.
    * [Statistics by hyper-parameter value](#stats_max_depth)<a href='#stats_max_depth'></a>.
    * [Describing hyper-parameter values](#describing_max_depth_values)<a href='#describing_max_depth_values'></a>.
    * [Data visualization](#data_vis_max_depth)<a href='#data_vis_max_depth'></a>.
<br>
<br>
6. [Learning rate](#learning_rate)<a href='#learning_rate'></a>.
    * [Processing data](#proc_data_learning_rate)<a href='#proc_data_learning_rate'></a>.
    * [Statistics by hyper-parameter value](#stats_learning_rate)<a href='#stats_learning_rate'></a>.
    * [Describing hyper-parameter values](#describing_learning_rate_values)<a href='#describing_learning_rate_values'></a>.
    * [Data visualization](#data_vis_learning_rate)<a href='#data_vis_learning_rate'></a>.
<br>
<br>
7. [Number of estimators](#n_estimators)<a href='#n_estimators'></a>.
    * [Processing data](#proc_data_n_estimators)<a href='#proc_data_n_estimators'></a>.
    * [Statistics by hyper-parameter value](#stats_n_estimators)<a href='#stats_n_estimators'></a>.
    * [Describing hyper-parameter values](#describing_n_estimators_values)<a href='#describing_n_estimators_values'></a>.
    * [Data visualization](#data_vis_n_estimators)<a href='#data_vis_n_estimators'></a>.
<br>
<br>
8. [Appendix: grid search with ROC-AUC and average precision score](#grid_search)<a href='#grid_search'></a>.

<a id='libraries'></a>

## Libraries


```python
import pandas as pd
import numpy as np
import os
import json

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import cufflinks as cf
init_notebook_mode(connected=True)
# cf.go_offline()
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        


<a id='functions'></a>

## Functions


```python
# Function that indicates the percentile of a value for a given variable:
def percentile_cut(x, p=10):
    intervals_ref = pd.qcut(x, q=p).unique().sort_values()

    intervals = []
    for i in range(len(intervals_ref)):
        if i==0:
            intervals.append(pd.Interval(intervals_ref[i].left - 0.001, intervals_ref[i].right))
        elif i==(len(intervals_ref) - 1):
            intervals.append(pd.Interval(intervals_ref[i].left, intervals_ref[i].right + 0.001))
        else:
            intervals.append(intervals_ref[i])
    
    interval = []
    percentile = []
    
    for j in range(len(x)):
        for k in range(len(intervals)):
            if x.iloc[j] in intervals[k]:
                interval.append(intervals[k])
                percentile.append(int(k + 1))
    
    return {'interval': interval, 'percentile': percentile}
```

<a id='imports'></a>

## Importing data

### Performance metrics


```python
# Results of tests for subsample:
with open('../Datasets/tun_subsample.json') as json_file:
    tun_subsample = json.load(json_file)

# Results of tests for max depth:
tun_max_depth = {}
with open('../Datasets/tun_max_depth_500.json') as json_file:
    tun_max_depth['500'] = json.load(json_file)

with open('../Datasets/tun_max_depth_1000.json') as json_file:
    tun_max_depth['1000'] = json.load(json_file)

# Results of tests for learning rate:
tun_learning_rate = {}
with open('../Datasets/tun_learning_rate_II100.json') as json_file:
    tun_learning_rate['II100'] = json.load(json_file)
    
with open('../Datasets/tun_learning_rate_II250.json') as json_file:
    tun_learning_rate['II250'] = json.load(json_file)

with open('../Datasets/tun_learning_rate_I500.json') as json_file:
    tun_learning_rate['I500'] = json.load(json_file)

with open('../Datasets/tun_learning_rate_II500.json') as json_file:
    tun_learning_rate['II500'] = json.load(json_file)

# Results of tests for number of estimators:
tun_n_estimators = {}
with open('../Datasets/tun_n_estimators_005.json') as json_file:
    tun_n_estimators['005'] = json.load(json_file)

with open('../Datasets/tun_n_estimators_01.json') as json_file:
    tun_n_estimators['01'] = json.load(json_file)
```

### Dataset information


```python
stores = []
n_orders = []
n_vars = []
avg_y = []

# Additional datasets information:
with open('../Datasets/data_info_dict.json') as json_file:
    data_info_dict = json.load(json_file)

# Loop over datasets:
for s in tun_subsample.keys():
    stores.append(int(s))
    n_orders.append(int(tun_subsample[s]['n_orders']))
    n_vars.append(int(tun_subsample[s]['n_vars']))
    avg_y.append(data_info_dict[s]['avg_y'])

data_info = pd.DataFrame(data={'store_id': stores, 'n_orders': n_orders, 'n_vars': n_vars, 'avg_y': avg_y})
print('\033[1mShape of data_info:\033[0m ' + str(data_info.shape) + '.')
data_info.head()
```

    [1mShape of data_info:[0m (100, 4).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>n_orders</th>
      <th>n_vars</th>
      <th>avg_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11729</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10311</td>
      <td>1999</td>
      <td>2534</td>
      <td>0.014007</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7988</td>
      <td>1077</td>
      <td>3051</td>
      <td>0.082637</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4736</td>
      <td>945</td>
      <td>2393</td>
      <td>0.048677</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3481</td>
      <td>4378</td>
      <td>2545</td>
      <td>0.013020</td>
    </tr>
  </tbody>
</table>
</div>



<a id='subsample'></a>

## Subsample

Main findings are listed below:
* Better performance metrics for $subsample < 1$.
* The introduction of randomness ($subsample < 1$) is particularly promising for small and more balanced datasets.
* Positive and concave relationship between dataset length and predictive performance.

<a id='proc_data_subsample'></a>

### Processing data

#### Performance metrics


```python
# Assessing missing hyper-parameter values:
for s in tun_subsample.keys():
    for m in ['test_roc_auc', 'test_prec_avg', 'test_pr_auc', 'test_deviance', 'test_brier_score']:
        if len(tun_subsample[s][m].keys()) != 5:
            print('Missing hyper-parameter value for store ' + str(s) + ' and metric ' + m + '!')
```


```python
# Collecting reference data:
param_value = []
stores = []

# Loop over datasets:
for s in tun_subsample.keys():
    # Loop over hyper-parameter values:
    for v in tun_subsample[s]['test_roc_auc'].keys():
        param_value.append(float(v))
        stores.append(int(s))

metrics_subsample = pd.DataFrame(data=param_value, columns=['param_value'], index=stores)

# Collecting performance metrics:
for m in ['test_roc_auc', 'test_prec_avg', 'test_pr_auc', 'test_deviance', 'test_brier_score']:
    stores = []
    param_value = []
    ref = []
    
    # Loop over datasets:
    for s in tun_subsample.keys():
        # Loop over hyper-parameter values:
        for v in tun_subsample[s][m].keys():
            stores.append(int(s))
            ref.append(float(tun_subsample[s][m][v]))

    metrics_subsample = pd.concat([metrics_subsample, pd.DataFrame(data={m: ref},
                                                                   index=stores)], axis=1)

metrics_subsample.index.name = 'store_id'
metrics_subsample.reset_index(inplace=True, drop=False)
print('\033[1mShape of metrics_subsample:\033[0m ' + str(metrics_subsample.shape) + '.')
metrics_subsample.head()
```

    [1mShape of metrics_subsample:[0m (500, 7).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>param_value</th>
      <th>test_roc_auc</th>
      <th>test_prec_avg</th>
      <th>test_pr_auc</th>
      <th>test_deviance</th>
      <th>test_brier_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11729</td>
      <td>0.50</td>
      <td>0.797171</td>
      <td>0.198029</td>
      <td>0.192070</td>
      <td>884.605920</td>
      <td>0.043620</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11729</td>
      <td>0.75</td>
      <td>0.801276</td>
      <td>0.191084</td>
      <td>0.184379</td>
      <td>885.512767</td>
      <td>0.043436</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11729</td>
      <td>0.80</td>
      <td>0.794766</td>
      <td>0.199539</td>
      <td>0.192724</td>
      <td>885.571017</td>
      <td>0.043495</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11729</td>
      <td>0.90</td>
      <td>0.778057</td>
      <td>0.175046</td>
      <td>0.168938</td>
      <td>886.412812</td>
      <td>0.043787</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11729</td>
      <td>1.00</td>
      <td>0.771984</td>
      <td>0.165714</td>
      <td>0.154823</td>
      <td>886.550523</td>
      <td>0.041960</td>
    </tr>
  </tbody>
</table>
</div>



<a id='stats_subsample'></a>

### Statistics by hyper-parameter value

#### Basic statistics for each performance metric


```python
# Test ROC-AUC:
metrics_subsample.groupby('param_value').describe()[['test_roc_auc']].sort_values(('test_roc_auc','mean'),
                                                                                  ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_roc_auc</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.80</th>
      <td>97.0</td>
      <td>0.850854</td>
      <td>0.133542</td>
      <td>0.462538</td>
      <td>0.804943</td>
      <td>0.898334</td>
      <td>0.941743</td>
      <td>0.998088</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>97.0</td>
      <td>0.849821</td>
      <td>0.131355</td>
      <td>0.497642</td>
      <td>0.780769</td>
      <td>0.901462</td>
      <td>0.948435</td>
      <td>0.998566</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>97.0</td>
      <td>0.848756</td>
      <td>0.131847</td>
      <td>0.464623</td>
      <td>0.799521</td>
      <td>0.895415</td>
      <td>0.943075</td>
      <td>0.983789</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>97.0</td>
      <td>0.831475</td>
      <td>0.145146</td>
      <td>0.348914</td>
      <td>0.780703</td>
      <td>0.881524</td>
      <td>0.930948</td>
      <td>0.990440</td>
    </tr>
    <tr>
      <th>1.00</th>
      <td>97.0</td>
      <td>0.826238</td>
      <td>0.158423</td>
      <td>0.342522</td>
      <td>0.756487</td>
      <td>0.888135</td>
      <td>0.946254</td>
      <td>0.987605</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test average precision score:
metrics_subsample.groupby('param_value').describe()[['test_prec_avg']].sort_values(('test_prec_avg','mean'),
                                                                                   ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_prec_avg</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.90</th>
      <td>97.0</td>
      <td>0.430315</td>
      <td>0.264026</td>
      <td>0.001986</td>
      <td>0.189474</td>
      <td>0.436735</td>
      <td>0.582394</td>
      <td>0.955605</td>
    </tr>
    <tr>
      <th>0.80</th>
      <td>97.0</td>
      <td>0.423809</td>
      <td>0.265607</td>
      <td>0.001986</td>
      <td>0.199539</td>
      <td>0.425727</td>
      <td>0.590697</td>
      <td>0.951808</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>97.0</td>
      <td>0.420588</td>
      <td>0.270484</td>
      <td>0.001986</td>
      <td>0.191084</td>
      <td>0.436735</td>
      <td>0.625278</td>
      <td>0.963963</td>
    </tr>
    <tr>
      <th>1.00</th>
      <td>97.0</td>
      <td>0.412980</td>
      <td>0.273139</td>
      <td>0.000956</td>
      <td>0.188237</td>
      <td>0.398212</td>
      <td>0.635556</td>
      <td>0.952785</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>97.0</td>
      <td>0.401102</td>
      <td>0.274657</td>
      <td>0.001986</td>
      <td>0.146101</td>
      <td>0.423420</td>
      <td>0.629219</td>
      <td>0.950403</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test Brier score:
metrics_subsample.groupby('param_value').describe()[['test_brier_score']].sort_values(('test_brier_score','mean'),
                                                                                      ascending=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_brier_score</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.50</th>
      <td>97.0</td>
      <td>0.032109</td>
      <td>0.031915</td>
      <td>0.001986</td>
      <td>0.011161</td>
      <td>0.019212</td>
      <td>0.039894</td>
      <td>0.174897</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>97.0</td>
      <td>0.032244</td>
      <td>0.033646</td>
      <td>0.001986</td>
      <td>0.010485</td>
      <td>0.018015</td>
      <td>0.038738</td>
      <td>0.179383</td>
    </tr>
    <tr>
      <th>0.80</th>
      <td>97.0</td>
      <td>0.032586</td>
      <td>0.033537</td>
      <td>0.001851</td>
      <td>0.010825</td>
      <td>0.018246</td>
      <td>0.039045</td>
      <td>0.179733</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>97.0</td>
      <td>0.032766</td>
      <td>0.035044</td>
      <td>0.001746</td>
      <td>0.010505</td>
      <td>0.017403</td>
      <td>0.039367</td>
      <td>0.198253</td>
    </tr>
    <tr>
      <th>1.00</th>
      <td>97.0</td>
      <td>0.035698</td>
      <td>0.042861</td>
      <td>0.001986</td>
      <td>0.010248</td>
      <td>0.018445</td>
      <td>0.039518</td>
      <td>0.216279</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test binomial deviance:
metrics_subsample.groupby('param_value').describe()[['test_deviance']].sort_values(('test_deviance','mean'),
                                                                                   ascending=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_deviance</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.75</th>
      <td>97.0</td>
      <td>2426.769644</td>
      <td>3871.417738</td>
      <td>53.542817</td>
      <td>343.073251</td>
      <td>950.301493</td>
      <td>2688.006956</td>
      <td>22106.123339</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>97.0</td>
      <td>2426.853649</td>
      <td>3870.969786</td>
      <td>53.513504</td>
      <td>343.702973</td>
      <td>948.606230</td>
      <td>2687.229826</td>
      <td>22105.818028</td>
    </tr>
    <tr>
      <th>0.80</th>
      <td>97.0</td>
      <td>2427.052235</td>
      <td>3871.342180</td>
      <td>53.508029</td>
      <td>343.443048</td>
      <td>948.664086</td>
      <td>2690.367984</td>
      <td>22103.846930</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>97.0</td>
      <td>2427.355443</td>
      <td>3872.225124</td>
      <td>54.753862</td>
      <td>342.841624</td>
      <td>950.656932</td>
      <td>2693.724586</td>
      <td>22108.276292</td>
    </tr>
    <tr>
      <th>1.00</th>
      <td>97.0</td>
      <td>2428.244825</td>
      <td>3870.559994</td>
      <td>53.548208</td>
      <td>344.590371</td>
      <td>948.777849</td>
      <td>2686.686159</td>
      <td>22107.145452</td>
    </tr>
  </tbody>
</table>
</div>



<a id='reference1'></a>
#### Averages of performance metrics by hyper-parameter value


```python
metrics_subsample.groupby('param_value').mean().sort_values('test_roc_auc', ascending=False).drop('store_id',
                                                                                                  axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test_roc_auc</th>
      <th>test_prec_avg</th>
      <th>test_pr_auc</th>
      <th>test_deviance</th>
      <th>test_brier_score</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.80</th>
      <td>0.850854</td>
      <td>0.423809</td>
      <td>0.428852</td>
      <td>2427.052235</td>
      <td>0.032586</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>0.849821</td>
      <td>0.430315</td>
      <td>0.436895</td>
      <td>2426.853649</td>
      <td>0.032766</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.848756</td>
      <td>0.420588</td>
      <td>0.428653</td>
      <td>2426.769644</td>
      <td>0.032244</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>0.831475</td>
      <td>0.401102</td>
      <td>0.407135</td>
      <td>2427.355443</td>
      <td>0.032109</td>
    </tr>
    <tr>
      <th>1.00</th>
      <td>0.826238</td>
      <td>0.412980</td>
      <td>0.423829</td>
      <td>2428.244825</td>
      <td>0.035698</td>
    </tr>
  </tbody>
</table>
</div>



[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

#### Frequency of best hyper-parameter values


```python
best_subsample_values = metrics_subsample.groupby('store_id').idxmax()['test_roc_auc'].values
print('\033[1mRelative frequency of highest performance metric by hyper-parameter value:\033[0m')
print(metrics_subsample.reindex(best_subsample_values).param_value.value_counts()/len(best_subsample_values))
```

    [1mRelative frequency of highest performance metric by hyper-parameter value:[0m
    1.00    0.23
    0.90    0.20
    0.75    0.20
    0.50    0.18
    0.80    0.16
    Name: param_value, dtype: float64
    

#### Average performance metric by best hyper-parameter value


```python
# Dataframe with best hyper-parameter value by dataset:
best_subsample_values = metrics_subsample.reindex(best_subsample_values)[['store_id', 'param_value', 'test_roc_auc']]
best_subsample_values = best_subsample_values.merge(data_info, on='store_id', how='inner')
print('\033[1mShape of best_subsample_values:\033[0m ' + str(best_subsample_values.shape) + '.')
best_subsample_values.head()
```

    [1mShape of best_subsample_values:[0m (97, 6).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>param_value</th>
      <th>test_roc_auc</th>
      <th>n_orders</th>
      <th>n_vars</th>
      <th>avg_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>720.0</td>
      <td>0.75</td>
      <td>0.853819</td>
      <td>4028</td>
      <td>1858</td>
      <td>0.011668</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1098.0</td>
      <td>1.00</td>
      <td>0.964134</td>
      <td>19152</td>
      <td>4026</td>
      <td>0.023705</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1181.0</td>
      <td>1.00</td>
      <td>0.981234</td>
      <td>3467</td>
      <td>2698</td>
      <td>0.033458</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1210.0</td>
      <td>0.50</td>
      <td>0.500000</td>
      <td>4028</td>
      <td>2101</td>
      <td>0.001490</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1241.0</td>
      <td>0.75</td>
      <td>0.900240</td>
      <td>206</td>
      <td>3791</td>
      <td>0.320388</td>
    </tr>
  </tbody>
</table>
</div>



<a id='reference5'></a>


```python
best_subsample_values.groupby('param_value').mean().sort_values('test_roc_auc', ascending=False)[['test_roc_auc']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test_roc_auc</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.00</th>
      <td>0.912122</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.877401</td>
    </tr>
    <tr>
      <th>0.80</th>
      <td>0.870712</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>0.863987</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>0.801953</td>
    </tr>
  </tbody>
</table>
</div>



[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

<a id='describing_subsample_values'></a>

### Describing hyper-parameter values

<a id='reference6'></a>
#### Average numbers of observations and features by best hyper-parameter value


```python
best_subsample_values.groupby('param_value').mean()[['n_orders']].sort_values('n_orders', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_orders</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.00</th>
      <td>16303.173913</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>7136.400000</td>
    </tr>
    <tr>
      <th>0.80</th>
      <td>4128.062500</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>2843.666667</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>2638.700000</td>
    </tr>
  </tbody>
</table>
</div>




```python
best_subsample_values.groupby('param_value').mean()[['n_vars']].sort_values('n_vars', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_vars</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.00</th>
      <td>2425.478261</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>2405.388889</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>2387.100000</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>2375.800000</td>
    </tr>
    <tr>
      <th>0.80</th>
      <td>2191.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Average of response variable by best hyper-parameter value


```python
best_subsample_values.groupby('param_value').mean()[['avg_y']].sort_values('avg_y', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_y</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.75</th>
      <td>0.084155</td>
    </tr>
    <tr>
      <th>0.80</th>
      <td>0.074008</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>0.041817</td>
    </tr>
    <tr>
      <th>1.00</th>
      <td>0.030399</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>0.023904</td>
    </tr>
  </tbody>
</table>
</div>



[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

#### Most frequent best hyper-parameter values by quartile of number of observations


```python
best_subsample_values['quartile_n_orders'] = percentile_cut(best_subsample_values.n_orders, p=4)['percentile']

print('\033[1mFrequency of best hyper-parameter values by quartile of number of observations:\033[0m')
for q in range(1,5):
    print('\033[1mNumber of orders in ' +
          str(np.sort(np.unique(percentile_cut(best_subsample_values.n_orders, p=4)['interval']))[q-1]) +
          ' (quartile ' + str(q) + ')\033[0m:')
    print(best_subsample_values[best_subsample_values.quartile_n_orders==q].param_value.value_counts())
    print('\n')
```

    [1mFrequency of best hyper-parameter values by quartile of number of observations:[0m
    [1mNumber of orders in (156.998, 999.0] (quartile 1)[0m:
    0.50    10
    0.75     7
    0.80     6
    0.90     2
    Name: param_value, dtype: int64
    
    
    [1mNumber of orders in (999.0, 2812.0] (quartile 2)[0m:
    0.75    7
    0.90    6
    0.80    5
    1.00    4
    0.50    2
    Name: param_value, dtype: int64
    
    
    [1mNumber of orders in (2812.0, 7946.0] (quartile 3)[0m:
    1.00    8
    0.90    6
    0.75    5
    0.50    3
    0.80    2
    Name: param_value, dtype: int64
    
    
    [1mNumber of orders in (7946.0, 63963.001] (quartile 4)[0m:
    1.00    11
    0.90     6
    0.80     3
    0.50     3
    0.75     1
    Name: param_value, dtype: int64
    
    
    

#### Most frequent best hyper-parameter values by quartile of number of features


```python
best_subsample_values['quartile_n_vars'] = percentile_cut(best_subsample_values.n_vars, p=4)['percentile']

print('\033[1mFrequency of best hyper-parameter values by quartile of number of features:\033[0m')
for q in range(1,5):
    print('\033[1mNumber of vars in ' +
          str(np.sort(np.unique(percentile_cut(best_subsample_values.n_vars, p=4)['interval']))[q-1]) +
          ' (quartile ' + str(q) + ')\033[0m:')
    print(best_subsample_values[best_subsample_values.quartile_n_vars==q].param_value.value_counts())
    print('\n')
```

    [1mFrequency of best hyper-parameter values by quartile of number of features:[0m
    [1mNumber of vars in (1415.998, 2069.0] (quartile 1)[0m:
    0.80    7
    1.00    6
    0.90    5
    0.75    4
    0.50    3
    Name: param_value, dtype: int64
    
    
    [1mNumber of vars in (2069.0, 2321.0] (quartile 2)[0m:
    0.75    6
    0.50    6
    0.90    5
    1.00    5
    0.80    2
    Name: param_value, dtype: int64
    
    
    [1mNumber of vars in (2321.0, 2534.0] (quartile 3)[0m:
    0.80    5
    0.90    5
    0.50    5
    1.00    5
    0.75    4
    Name: param_value, dtype: int64
    
    
    [1mNumber of vars in (2534.0, 4026.001] (quartile 4)[0m:
    1.00    7
    0.75    6
    0.90    5
    0.50    4
    0.80    2
    Name: param_value, dtype: int64
    
    
    

#### Correlation between performance metric of best hyper-parameter value and dataset information


```python
best_subsample_values[['test_roc_auc', 'n_orders', 'n_vars', 'avg_y']].corr()[['test_roc_auc']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test_roc_auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>test_roc_auc</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>n_orders</th>
      <td>0.262390</td>
    </tr>
    <tr>
      <th>n_vars</th>
      <td>0.148748</td>
    </tr>
    <tr>
      <th>avg_y</th>
      <td>0.145395</td>
    </tr>
  </tbody>
</table>
</div>



#### Correlation between performance metric and dataset information by hyper-parameter value


```python
metrics_subsample = metrics_subsample.merge(data_info, on='store_id', how='left')
print('\033[1mShape of metrics_subsample:\033[0m ' + str(metrics_subsample.shape) + '.')
metrics_subsample.head()
```

    [1mShape of metrics_subsample:[0m (500, 10).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>param_value</th>
      <th>test_roc_auc</th>
      <th>test_prec_avg</th>
      <th>test_pr_auc</th>
      <th>test_deviance</th>
      <th>test_brier_score</th>
      <th>n_orders</th>
      <th>n_vars</th>
      <th>avg_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11729</td>
      <td>0.50</td>
      <td>0.797171</td>
      <td>0.198029</td>
      <td>0.192070</td>
      <td>884.605920</td>
      <td>0.043620</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11729</td>
      <td>0.75</td>
      <td>0.801276</td>
      <td>0.191084</td>
      <td>0.184379</td>
      <td>885.512767</td>
      <td>0.043436</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11729</td>
      <td>0.80</td>
      <td>0.794766</td>
      <td>0.199539</td>
      <td>0.192724</td>
      <td>885.571017</td>
      <td>0.043495</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11729</td>
      <td>0.90</td>
      <td>0.778057</td>
      <td>0.175046</td>
      <td>0.168938</td>
      <td>886.412812</td>
      <td>0.043787</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11729</td>
      <td>1.00</td>
      <td>0.771984</td>
      <td>0.165714</td>
      <td>0.154823</td>
      <td>886.550523</td>
      <td>0.041960</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
  </tbody>
</table>
</div>




```python
for v in metrics_subsample.param_value.unique():
    print('\033[1msubsample = ' + str(v) + '\033[0m')
    print(metrics_subsample[metrics_subsample.param_value==v][['test_roc_auc',
                                                               'n_orders', 'n_vars',
                                                               'avg_y']].corr()[['test_roc_auc']])
    print('\n')
```

    [1msubsample = 0.5[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.227759
    n_vars            0.132379
    avg_y             0.226679
    
    
    [1msubsample = 0.75[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.254695
    n_vars            0.115074
    avg_y             0.185854
    
    
    [1msubsample = 0.8[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.236481
    n_vars            0.099442
    avg_y             0.170486
    
    
    [1msubsample = 0.9[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.292085
    n_vars            0.134803
    avg_y             0.151452
    
    
    [1msubsample = 1.0[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.303740
    n_vars            0.052553
    avg_y             0.169682
    
    
    

<a id='data_vis_subsample'></a>

### Data visualization

#### Average of performance metric by hyper-parameter value


```python
# Select a performance metric:
metric = 'test_roc_auc'

fig=px.scatter(x=metrics_subsample['param_value'].apply(lambda x: 'eta = ' + str(x)).unique(),
               y=metrics_subsample.groupby('param_value').mean()[metric], 
               error_y=np.array(metrics_subsample.groupby('param_value').std()[metric]),
               color_discrete_sequence=['#0b6fab'],
               width=900, height=500,
               title='Average of ' + metric + ' by subsample value',
               labels={'y': metric, 'x': ''})

fig.add_trace(
    go.Scatter(
        x=metrics_subsample['param_value'].apply(lambda x: 'eta = ' + str(x)).unique(),
        y=metrics_subsample.groupby('param_value').mean()[metric],
        line = dict(color='#0b6fab', width=2, dash='dash'),
        name='avg_' + metric
              )
)
```


<div>                            <div id="c6268a4e-182a-44b7-88da-da72db342532" class="plotly-graph-div" style="height:500px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("c6268a4e-182a-44b7-88da-da72db342532")) {                    Plotly.newPlot(                        "c6268a4e-182a-44b7-88da-da72db342532",                        [{"error_y": {"array": [0.14514574907214406, 0.1318472947132773, 0.13354163484375595, 0.13135478535515036, 0.15842320628834616]}, "hovertemplate": "=%{x}<br>test_roc_auc=%{y}<extra></extra>", "legendgroup": "", "marker": {"color": "#0b6fab", "symbol": "circle"}, "mode": "markers", "name": "", "orientation": "v", "showlegend": false, "type": "scatter", "x": ["eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0"], "xaxis": "x", "y": [0.8314750274053093, 0.848755951561052, 0.8508535435654355, 0.8498207221430866, 0.826237798285497], "yaxis": "y"}, {"line": {"color": "#0b6fab", "dash": "dash", "width": 2}, "name": "avg_test_roc_auc", "type": "scatter", "x": ["eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0"], "y": [0.8314750274053093, 0.848755951561052, 0.8508535435654355, 0.8498207221430866, 0.826237798285497]}],                        {"height": 500, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Average of test_roc_auc by subsample value"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": ""}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "test_roc_auc"}}},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('c6268a4e-182a-44b7-88da-da72db342532');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


<a id='reference2'></a>
#### Boxplot of performance metric by hyper-parameter value


```python
# Select a performance metric:
metric = 'test_roc_auc'

px.box(data_frame=metrics_subsample,
       x=metrics_subsample['param_value'].apply(lambda x: 'eta = ' + str(x)),
       y=metric, hover_data=['store_id'],
       color_discrete_sequence=['#0b6fab'],
       width=900, height=500,
       labels={'x': ' '},
       title='Distribution of ' + metric + ' by subsample value')
```


<div>                            <div id="bf64966e-cceb-4ae2-bd20-d8192297081b" class="plotly-graph-div" style="height:500px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("bf64966e-cceb-4ae2-bd20-d8192297081b")) {                    Plotly.newPlot(                        "bf64966e-cceb-4ae2-bd20-d8192297081b",                        [{"alignmentgroup": "True", "customdata": [[11729], [11729], [11729], [11729], [11729], [10311], [10311], [10311], [10311], [10311], [7988], [7988], [7988], [7988], [7988], [4736], [4736], [4736], [4736], [4736], [3481], [3481], [3481], [3481], [3481], [4838], [4838], [4838], [4838], [4838], [5848], [5848], [5848], [5848], [5848], [6106], [6106], [6106], [6106], [6106], [1559], [1559], [1559], [1559], [1559], [5342], [5342], [5342], [5342], [5342], [3781], [3781], [3781], [3781], [3781], [4408], [4408], [4408], [4408], [4408], [7292], [7292], [7292], [7292], [7292], [6044], [6044], [6044], [6044], [6044], [8181], [8181], [8181], [8181], [8181], [2352], [2352], [2352], [2352], [2352], [9491], [9491], [9491], [9491], [9491], [5847], [5847], [5847], [5847], [5847], [7939], [7939], [7939], [7939], [7939], [6078], [6078], [6078], [6078], [6078], [10268], [10268], [10268], [10268], [10268], [10060], [10060], [10060], [10060], [10060], [6256], [6256], [6256], [6256], [6256], [8436], [8436], [8436], [8436], [8436], [5085], [5085], [5085], [5085], [5085], [8783], [8783], [8783], [8783], [8783], [6047], [6047], [6047], [6047], [6047], [8832], [8832], [8832], [8832], [8832], [1961], [1961], [1961], [1961], [1961], [7845], [7845], [7845], [7845], [7845], [2699], [2699], [2699], [2699], [2699], [6004], [6004], [6004], [6004], [6004], [2868], [2868], [2868], [2868], [2868], [1875], [1875], [1875], [1875], [1875], [5593], [5593], [5593], [5593], [5593], [7849], [7849], [7849], [7849], [7849], [11223], [11223], [11223], [11223], [11223], [6170], [6170], [6170], [6170], [6170], [5168], [5168], [5168], [5168], [5168], [2866], [2866], [2866], [2866], [2866], [3437], [3437], [3437], [3437], [3437], [6929], [6929], [6929], [6929], [6929], [8894], [8894], [8894], [8894], [8894], [9177], [9177], [9177], [9177], [9177], [10349], [10349], [10349], [10349], [10349], [3988], [3988], [3988], [3988], [3988], [1549], [1549], [1549], [1549], [1549], [9541], [9541], [9541], [9541], [9541], [1181], [1181], [1181], [1181], [1181], [7790], [7790], [7790], [7790], [7790], [5663], [5663], [5663], [5663], [5663], [4601], [4601], [4601], [4601], [4601], [1603], [1603], [1603], [1603], [1603], [2212], [2212], [2212], [2212], [2212], [9761], [9761], [9761], [9761], [9761], [5428], [5428], [5428], [5428], [5428], [1098], [1098], [1098], [1098], [1098], [5939], [5939], [5939], [5939], [5939], [7333], [7333], [7333], [7333], [7333], [8358], [8358], [8358], [8358], [8358], [10650], [10650], [10650], [10650], [10650], [6083], [6083], [6083], [6083], [6083], [1424], [1424], [1424], [1424], [1424], [9281], [9281], [9281], [9281], [9281], [7161], [7161], [7161], [7161], [7161], [7185], [7185], [7185], [7185], [7185], [12980], [12980], [12980], [12980], [12980], [8282], [8282], [8282], [8282], [8282], [3962], [3962], [3962], [3962], [3962], [720], [720], [720], [720], [720], [11723], [11723], [11723], [11723], [11723], [8446], [8446], [8446], [8446], [8446], [8790], [8790], [8790], [8790], [8790], [1241], [1241], [1241], [1241], [1241], [9098], [9098], [9098], [9098], [9098], [1739], [1739], [1739], [1739], [1739], [4636], [4636], [4636], [4636], [4636], [8421], [8421], [8421], [8421], [8421], [5215], [5215], [5215], [5215], [5215], [7062], [7062], [7062], [7062], [7062], [6714], [6714], [6714], [6714], [6714], [7630], [7630], [7630], [7630], [7630], [6970], [6970], [6970], [6970], [6970], [3146], [3146], [3146], [3146], [3146], [5860], [5860], [5860], [5860], [5860], [7755], [7755], [7755], [7755], [7755], [6105], [6105], [6105], [6105], [6105], [4030], [4030], [4030], [4030], [4030], [3859], [3859], [3859], [3859], [3859], [4268], [4268], [4268], [4268], [4268], [9409], [9409], [9409], [9409], [9409], [1979], [1979], [1979], [1979], [1979], [12658], [12658], [12658], [12658], [12658], [5394], [5394], [5394], [5394], [5394], [1210], [1210], [1210], [1210], [1210], [6971], [6971], [6971], [6971], [6971], [2056], [2056], [2056], [2056], [2056], [2782], [2782], [2782], [2782], [2782], [4974], [4974], [4974], [4974], [4974], [6966], [6966], [6966], [6966], [6966]], "hovertemplate": " =%{x}<br>test_roc_auc=%{y}<br>store_id=%{customdata[0]}<extra></extra>", "legendgroup": "", "marker": {"color": "#0b6fab"}, "name": "", "notched": false, "offsetgroup": "", "orientation": "v", "showlegend": false, "type": "box", "x": ["eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0", "eta = 0.5", "eta = 0.75", "eta = 0.8", "eta = 0.9", "eta = 1.0"], "x0": " ", "xaxis": "x", "y": [0.7971710449843077, 0.8012757177728699, 0.7947663605718935, 0.7780570731140299, 0.7719836103684761, 0.5545614919354839, 0.7995211693548386, 0.8469632056451614, 0.8589969758064516, 0.8397807459677419, 0.5423783287419651, 0.5636363636363636, 0.5420569329660239, 0.5356290174471994, 0.5604683195592286, 0.9182119205298014, 0.9212472406181016, 0.9161699779249448, 0.9123068432671082, 0.8881346578366445, 0.7770484671682276, 0.7293106095501306, 0.7353626081170992, 0.7551947387276728, 0.7859921183274476, 0.9373512713250606, 0.9637087888373794, 0.9696420501653488, 0.9708336929974662, 0.9722157462529042, 0.9711962833914053, 0.9602787456445993, 0.9667828106852497, 0.9714285714285715, 0.9679442508710802, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7122153209109732, 0.9296296296296296, 0.9401234567901234, 0.9384259259259259, 0.9333333333333333, 0.9265432098765433, 0.9293236344786395, 0.9421674799130033, 0.937440133424988, 0.9408321031339096, 0.934484837557331, 0.911489062342016, 0.9637564137070692, 0.9658391329052545, 0.9679752551598033, 0.9686787514223014, 0.7220052083333334, 0.8130208333333332, 0.8139322916666665, 0.815234375, 0.7630208333333333, 0.3797169811320755, 0.464622641509434, 0.4658018867924528, 0.4976415094339623, 0.49410377358490565, 0.9328715410430508, 0.9563418075366722, 0.941742560107131, 0.9631563073135956, 0.9714646418486316, 0.4948104693140794, 0.49616425992779783, 0.49526173285198555, 0.49909747292418777, 0.4749548736462094, 0.7560834955598816, 0.881755180138137, 0.8983339555721486, 0.9033374223312622, 0.8935654950798688, 0.9446806495584261, 0.938254742987414, 0.9489485771883361, 0.9470794562434254, 0.9501757163103295, 0.9662953628374056, 0.9709486312252154, 0.9702434586538168, 0.9713534525162038, 0.9727659741175549, 0.9281197742474916, 0.9252891583054625, 0.9213872630992198, 0.9220056438127091, 0.9183301978818283, 0.7600000000000001, 0.638888888888889, 0.6111111111111112, 0.6677777777777778, 0.48, 0.6431451612903226, 0.5873655913978495, 0.5782930107526881, 0.5514112903225806, 0.5413306451612904, 0.8665893679069605, 0.8710867502898704, 0.879413934858227, 0.8713678366888022, 0.8740030216787885, 0.8347402950733017, 0.8740075200827963, 0.9305247400951892, 0.9064420413937694, 0.9271555830951381, 0.6619098426246999, 0.7013870365430782, 0.7085889570552147, 0.7005868231528408, 0.702987463323553, 0.8745238095238095, 0.7783333333333333, 0.9004761904761904, 0.7854761904761903, 0.6878571428571426, 0.8324440132589538, 0.8636753173255719, 0.8849704907429865, 0.8780337941628265, 0.8745573611447975, 0.8815242081826489, 0.9016301270253921, 0.9131880542077966, 0.9177150229726773, 0.9280036614715383, 0.9289538714991762, 0.9413284409165793, 0.9264639808297139, 0.9332596974689232, 0.9260146772502621, 0.9131315987933636, 0.9103035444947211, 0.917020173453997, 0.9136500754147813, 0.9153940422322776, 0.7679738562091505, 0.7604166666666665, 0.8049428104575163, 0.764501633986928, 0.7317197712418301, 0.9568839623471452, 0.9701680302630421, 0.9682326031494677, 0.9716591888800915, 0.968223805753497, 0.8023818654523823, 0.8890566447946517, 0.8915161141290399, 0.8858619013253053, 0.8799329043479311, 0.9309480122324159, 0.9479434250764526, 0.9637308868501528, 0.9723318042813455, 0.9770107033639143, 0.9502151799687011, 0.943075117370892, 0.9305800078247262, 0.9216793818466356, 0.9462539123630673, 0.925487351374237, 0.9379563207846735, 0.9358429947429701, 0.9390341177715839, 0.9395720989309528, 0.7858541604434268, 0.8450330246606259, 0.8317484815294236, 0.8516462544392446, 0.8728467589365727, 0.7493960026356249, 0.7507870268687313, 0.7401713156160773, 0.7379749615638042, 0.7048100153744784, 0.9767149220313271, 0.9773147420853109, 0.9767895028316902, 0.9784699510781687, 0.9778225262579956, 0.777972027972028, 0.7316433566433567, 0.7447552447552448, 0.6258741258741258, 0.5305944055944056, 0.9425361927619013, 0.9426087824441853, 0.9464480088615251, 0.9481018670554868, 0.9450866033298422, 0.9616747181964573, 0.970048309178744, 0.9674718196457327, 0.9623188405797102, 0.949597423510467, 0.9045049130763417, 0.9215117157974301, 0.917838246409675, 0.9262131519274376, 0.9234467120181405, 0.8431196902271282, 0.8873509410699493, 0.8923308096861816, 0.901462174189447, 0.9035042092893333, 0.9113676398394409, 0.9347800976060527, 0.9363033873343153, 0.9325601374570447, 0.9308238759421295, 0.7907585004359198, 0.8181621960200101, 0.7878382610609356, 0.7880985432660973, 0.7916811267891757, 0.9785597572362279, 0.9837885154061625, 0.982983193277311, 0.9877684407096171, 0.9876050420168068, null, null, null, null, null, 0.8627402921953096, 0.9509227220299884, 0.9262351018838909, 0.9505094194540562, 0.9431949250288351, 0.9727111432808417, 0.9789430478073582, 0.9714797683779395, 0.9765553280135584, 0.9812336699385636, 0.9291074249605056, 0.8975118483412323, 0.8579186413902053, 0.8597946287519747, 0.6994470774091628, 0.9382198647032738, 0.9501893971244209, 0.9371661505045523, 0.948434829480631, 0.9543387563546651, 0.970843935538592, 0.9725800466497032, 0.971745122985581, 0.9697837150127226, 0.9677913839411931, 0.9688684913028447, 0.9705206390949744, 0.9704063395621856, 0.9712991275450538, 0.9735042347504584, 0.8849513688760806, 0.8804214697406341, 0.8810878962536024, 0.9125450288184438, 0.9349153458213255, 0.8170384889522452, 0.6065235210263721, 0.6225053456878118, 0.7123342836778332, 0.6768068424803992, 0.945488964419395, 0.9200189304306673, 0.9032719528460181, 0.9155767327797617, 0.905842619283225, 0.9146297287219569, 0.9358145527123277, 0.9573521716378858, 0.9631104858450875, 0.964134024275914, 0.7233754512635379, 0.9711191335740071, 0.9787906137184116, 0.9801444043321299, 0.9817238267148014, 0.949010535013427, 0.9534030159058047, 0.952968394959719, 0.9577508779177856, 0.9543334021896303, 0.9217071111882484, 0.9149915587181875, 0.9069210621814797, 0.9215611637924233, 0.8995309960712539, 0.8551335581606981, 0.8648760772977891, 0.8654381457095445, 0.8612627803650769, 0.8482281462448478, 0.6632474901789611, 0.6772846315622395, 0.8042934804174438, 0.658644498234197, 0.6121384072060633, 0.9546596500543565, 0.957997705980981, 0.9582107243826455, 0.9579004083867072, 0.9576081866179373, 0.87382079268489, 0.8765199566416605, 0.8529089796838275, 0.8679073161831783, 0.8519752728677678, 0.8910411622276029, 0.8621872477804681, 0.8405972558514931, 0.8317191283292978, 0.7610976594027442, 0.7807025710251515, 0.8110754239786497, 0.7924482924482924, 0.7807685146394824, 0.7661949758723953, 0.8948512585812357, 0.9121967963386728, 0.9162242562929063, 0.9128146453089244, 0.904096109839817, 0.9296311146752205, 0.9361467522052928, 0.9289127238706228, 0.9298483025928895, 0.9179697941726811, 0.9498580648945333, 0.9558748996625585, 0.9583512328021735, 0.9614807302025099, 0.9635393808397698, 0.8105516588733022, 0.8538187486083277, 0.8536656646626586, 0.8532203295479848, 0.83375083500334, 0.7853393271461717, 0.8867024361948954, 0.8952726218097448, 0.9261649265274554, 0.904741879350348, 0.9904397705544933, 0.9780114722753345, 0.9980879541108987, 0.9985659655831739, 0.4933078393881453, 0.8340004105090313, 0.7954125615763548, 0.814090722495895, 0.8204022988505748, 0.8185036945812808, 0.8938301282051282, 0.9002403846153846, 0.8998397435897436, 0.8810096153846154, 0.8565705128205128, 0.8898323972541677, 0.8954154408487118, 0.8893532138718017, 0.8978336453597218, 0.9008090398502274, 0.854261796042618, 0.8729071537290715, 0.873668188736682, 0.7423896499238964, 0.810882800608828, 0.5, 0.5, 0.5, 0.5, 0.5, 0.48556430446194226, 0.5253718285214348, 0.5332458442694663, 0.5328083989501313, 0.5336832895888014, 0.883217350179042, 0.9013242348489967, 0.8878060040988222, 0.8912967592279799, 0.8906211292029818, null, null, null, null, null, 0.3489138176638177, 0.5759437321937322, 0.6458110754985755, 0.6462562321937322, 0.5303151709401709, 0.8047619047619048, 0.8142857142857142, 0.8134920634920635, 0.7738095238095238, 0.6158730158730159, 0.9038074939656969, 0.9135766776707254, 0.9061001929632295, 0.9080499956297526, 0.9048698002460785, 0.6623417721518987, 0.7202531645569621, 0.7123417721518988, 0.7, 0.7564873417721518, null, null, null, null, null, 0.8609126984126984, 0.8699074074074074, 0.8657407407407407, 0.8704034391534391, 0.8617724867724869, 0.959812539374469, 0.9681700339841997, 0.9705716590570582, 0.9713039271255386, 0.972182648807715, 0.8744221315378812, 0.9022330815499696, 0.9073212545698314, 0.919102078494868, 0.886825698021144, 0.9600612114485587, 0.9644797026872498, 0.9648706824067534, 0.9663365060146416, 0.9662650366035493, 0.8721484438650806, 0.8807710064635272, 0.8809067948509044, 0.8751154201292706, 0.8691441257943621, 0.9548688516173661, 0.9687456934637928, 0.9656395831137844, 0.9650869954090635, 0.9560658997992308, 0.4173489810771471, 0.6120865680090571, 0.4625384117742196, 0.6760371179039302, 0.48982087983179695, 0.809375, 0.7875, 0.7760416666666667, 0.7921875, 0.7390625, 0.8951772737282536, 0.8389121338912133, 0.864787491741907, 0.8371504073992514, 0.8709535344637745, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6680351906158357, 0.7624633431085044, 0.7788856304985338, 0.7640762463343109, 0.3425219941348973, 0.948205937626192, 0.9545893128941073, 0.9596632031808157, 0.9611697418693504, 0.958515826421893, 0.7062305295950156, 0.679185112313494, 0.6831939662239712, 0.6847679947532382, 0.6811772421708476, 0.9122448979591836, 0.9081632653061225, 0.9081632653061225, 0.8785714285714286, 0.8760204081632653, 0.8151921253581005, 0.8163772977050985, 0.813499750221742, 0.8186915697289141, 0.8200181981302312], "y0": " ", "yaxis": "y"}],                        {"boxmode": "group", "height": 500, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Distribution of test_roc_auc by subsample value"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": " "}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "test_roc_auc"}}},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('bf64966e-cceb-4ae2-bd20-d8192297081b');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

<a id='reference3'></a>
#### Frequency of best hyper-parameter values


```python
plt.figure(figsize=(8,5))

best_param_freq = sns.countplot(best_subsample_values['param_value'], palette='Greens')

for p in best_param_freq.patches:
    best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

plt.ylabel('frequency')
plt.xlabel('subsample')
plt.title('Count of best subsample value', loc='left')
```




    Text(0.0, 1.0, 'Count of best subsample value')




![png](output_82_1.png)


[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

<a id='reference7'></a>


```python
# plt.figure(figsize=(12,5))

# best_param_freq = sns.countplot(best_subsample_values['param_value'], palette='Greens',
#                                 hue=best_subsample_values['quartile_n_orders'])

# for p in best_param_freq.patches:
#     best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                              ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

# plt.ylabel('frequency')
# plt.xlabel('subsample')
# # plt.legend(loc='upper center', title='quartile_n_orders')
# plt.legend(loc='upper left', title='quartile_n_orders', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
# plt.title('Count of best subsample value', loc='left')
```


```python
plt.figure(figsize=(12,5))

best_param_freq = sns.countplot(best_subsample_values['quartile_n_orders'], palette='Greens',
                                hue=best_subsample_values['param_value'])

for p in best_param_freq.patches:
    best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

plt.ylabel('frequency')
plt.xlabel('quartile_n_orders')
plt.legend(loc='upper left', title='subsample', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
plt.title('Count of best subsample value', loc='left')
```




    Text(0.0, 1.0, 'Count of best subsample value')




![png](output_86_1.png)


[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>


```python
# plt.figure(figsize=(12,5))

# best_param_freq = sns.countplot(best_subsample_values['param_value'], palette='Greens',
#                                 hue=best_subsample_values['quartile_n_vars'])

# for p in best_param_freq.patches:
#     best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                              ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

# plt.ylabel('frequency')
# plt.xlabel('subsample')
# plt.legend(loc='upper left', title='quartile_n_vars', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
# plt.title('Count of best subsample value', loc='left')
```


```python
plt.figure(figsize=(12,5))

best_param_freq = sns.countplot(best_subsample_values['quartile_n_vars'], palette='Greens',
                                hue=best_subsample_values['param_value'])

for p in best_param_freq.patches:
    best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

plt.ylabel('frequency')
plt.xlabel('quartile_n_vars')
plt.legend(loc='upper left', title='subsample', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
plt.title('Count of best subsample value', loc='left')
```




    Text(0.0, 1.0, 'Count of best subsample value')




![png](output_89_1.png)


<a id='performance_data_info'></a>

#### Performance metric against dataset information


```python
# Select a performance metric:
metric = 'test_roc_auc'

px.scatter(data_frame=metrics_subsample,
           x='n_orders', y=metric, hover_data=['store_id', 'param_value'],
           color_discrete_sequence=['#0b6fab'], trendline='ols',
           width=900, height=500, title='Relationship between ' + metric + ' and dataset sizes')
```


<div>                            <div id="11236549-ae15-4db9-bd07-568c7993f704" class="plotly-graph-div" style="height:500px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("11236549-ae15-4db9-bd07-568c7993f704")) {                    Plotly.newPlot(                        "11236549-ae15-4db9-bd07-568c7993f704",                        [{"customdata": [[11729.0, 0.5], [11729.0, 0.75], [11729.0, 0.8], [11729.0, 0.9], [11729.0, 1.0], [10311.0, 0.5], [10311.0, 0.75], [10311.0, 0.8], [10311.0, 0.9], [10311.0, 1.0], [7988.0, 0.5], [7988.0, 0.75], [7988.0, 0.8], [7988.0, 0.9], [7988.0, 1.0], [4736.0, 0.5], [4736.0, 0.75], [4736.0, 0.8], [4736.0, 0.9], [4736.0, 1.0], [3481.0, 0.5], [3481.0, 0.75], [3481.0, 0.8], [3481.0, 0.9], [3481.0, 1.0], [4838.0, 0.5], [4838.0, 0.75], [4838.0, 0.8], [4838.0, 0.9], [4838.0, 1.0], [5848.0, 0.5], [5848.0, 0.75], [5848.0, 0.8], [5848.0, 0.9], [5848.0, 1.0], [6106.0, 0.5], [6106.0, 0.75], [6106.0, 0.8], [6106.0, 0.9], [6106.0, 1.0], [1559.0, 0.5], [1559.0, 0.75], [1559.0, 0.8], [1559.0, 0.9], [1559.0, 1.0], [5342.0, 0.5], [5342.0, 0.75], [5342.0, 0.8], [5342.0, 0.9], [5342.0, 1.0], [3781.0, 0.5], [3781.0, 0.75], [3781.0, 0.8], [3781.0, 0.9], [3781.0, 1.0], [4408.0, 0.5], [4408.0, 0.75], [4408.0, 0.8], [4408.0, 0.9], [4408.0, 1.0], [7292.0, 0.5], [7292.0, 0.75], [7292.0, 0.8], [7292.0, 0.9], [7292.0, 1.0], [6044.0, 0.5], [6044.0, 0.75], [6044.0, 0.8], [6044.0, 0.9], [6044.0, 1.0], [8181.0, 0.5], [8181.0, 0.75], [8181.0, 0.8], [8181.0, 0.9], [8181.0, 1.0], [2352.0, 0.5], [2352.0, 0.75], [2352.0, 0.8], [2352.0, 0.9], [2352.0, 1.0], [9491.0, 0.5], [9491.0, 0.75], [9491.0, 0.8], [9491.0, 0.9], [9491.0, 1.0], [5847.0, 0.5], [5847.0, 0.75], [5847.0, 0.8], [5847.0, 0.9], [5847.0, 1.0], [7939.0, 0.5], [7939.0, 0.75], [7939.0, 0.8], [7939.0, 0.9], [7939.0, 1.0], [6078.0, 0.5], [6078.0, 0.75], [6078.0, 0.8], [6078.0, 0.9], [6078.0, 1.0], [10268.0, 0.5], [10268.0, 0.75], [10268.0, 0.8], [10268.0, 0.9], [10268.0, 1.0], [10060.0, 0.5], [10060.0, 0.75], [10060.0, 0.8], [10060.0, 0.9], [10060.0, 1.0], [6256.0, 0.5], [6256.0, 0.75], [6256.0, 0.8], [6256.0, 0.9], [6256.0, 1.0], [8436.0, 0.5], [8436.0, 0.75], [8436.0, 0.8], [8436.0, 0.9], [8436.0, 1.0], [5085.0, 0.5], [5085.0, 0.75], [5085.0, 0.8], [5085.0, 0.9], [5085.0, 1.0], [8783.0, 0.5], [8783.0, 0.75], [8783.0, 0.8], [8783.0, 0.9], [8783.0, 1.0], [6047.0, 0.5], [6047.0, 0.75], [6047.0, 0.8], [6047.0, 0.9], [6047.0, 1.0], [8832.0, 0.5], [8832.0, 0.75], [8832.0, 0.8], [8832.0, 0.9], [8832.0, 1.0], [1961.0, 0.5], [1961.0, 0.75], [1961.0, 0.8], [1961.0, 0.9], [1961.0, 1.0], [7845.0, 0.5], [7845.0, 0.75], [7845.0, 0.8], [7845.0, 0.9], [7845.0, 1.0], [2699.0, 0.5], [2699.0, 0.75], [2699.0, 0.8], [2699.0, 0.9], [2699.0, 1.0], [6004.0, 0.5], [6004.0, 0.75], [6004.0, 0.8], [6004.0, 0.9], [6004.0, 1.0], [2868.0, 0.5], [2868.0, 0.75], [2868.0, 0.8], [2868.0, 0.9], [2868.0, 1.0], [1875.0, 0.5], [1875.0, 0.75], [1875.0, 0.8], [1875.0, 0.9], [1875.0, 1.0], [5593.0, 0.5], [5593.0, 0.75], [5593.0, 0.8], [5593.0, 0.9], [5593.0, 1.0], [7849.0, 0.5], [7849.0, 0.75], [7849.0, 0.8], [7849.0, 0.9], [7849.0, 1.0], [11223.0, 0.5], [11223.0, 0.75], [11223.0, 0.8], [11223.0, 0.9], [11223.0, 1.0], [6170.0, 0.5], [6170.0, 0.75], [6170.0, 0.8], [6170.0, 0.9], [6170.0, 1.0], [5168.0, 0.5], [5168.0, 0.75], [5168.0, 0.8], [5168.0, 0.9], [5168.0, 1.0], [2866.0, 0.5], [2866.0, 0.75], [2866.0, 0.8], [2866.0, 0.9], [2866.0, 1.0], [3437.0, 0.5], [3437.0, 0.75], [3437.0, 0.8], [3437.0, 0.9], [3437.0, 1.0], [6929.0, 0.5], [6929.0, 0.75], [6929.0, 0.8], [6929.0, 0.9], [6929.0, 1.0], [8894.0, 0.5], [8894.0, 0.75], [8894.0, 0.8], [8894.0, 0.9], [8894.0, 1.0], [9177.0, 0.5], [9177.0, 0.75], [9177.0, 0.8], [9177.0, 0.9], [9177.0, 1.0], [10349.0, 0.5], [10349.0, 0.75], [10349.0, 0.8], [10349.0, 0.9], [10349.0, 1.0], [3988.0, 0.5], [3988.0, 0.75], [3988.0, 0.8], [3988.0, 0.9], [3988.0, 1.0], [1549.0, 0.5], [1549.0, 0.75], [1549.0, 0.8], [1549.0, 0.9], [1549.0, 1.0], [9541.0, 0.5], [9541.0, 0.75], [9541.0, 0.8], [9541.0, 0.9], [9541.0, 1.0], [1181.0, 0.5], [1181.0, 0.75], [1181.0, 0.8], [1181.0, 0.9], [1181.0, 1.0], [7790.0, 0.5], [7790.0, 0.75], [7790.0, 0.8], [7790.0, 0.9], [7790.0, 1.0], [5663.0, 0.5], [5663.0, 0.75], [5663.0, 0.8], [5663.0, 0.9], [5663.0, 1.0], [4601.0, 0.5], [4601.0, 0.75], [4601.0, 0.8], [4601.0, 0.9], [4601.0, 1.0], [1603.0, 0.5], [1603.0, 0.75], [1603.0, 0.8], [1603.0, 0.9], [1603.0, 1.0], [2212.0, 0.5], [2212.0, 0.75], [2212.0, 0.8], [2212.0, 0.9], [2212.0, 1.0], [9761.0, 0.5], [9761.0, 0.75], [9761.0, 0.8], [9761.0, 0.9], [9761.0, 1.0], [5428.0, 0.5], [5428.0, 0.75], [5428.0, 0.8], [5428.0, 0.9], [5428.0, 1.0], [1098.0, 0.5], [1098.0, 0.75], [1098.0, 0.8], [1098.0, 0.9], [1098.0, 1.0], [5939.0, 0.5], [5939.0, 0.75], [5939.0, 0.8], [5939.0, 0.9], [5939.0, 1.0], [7333.0, 0.5], [7333.0, 0.75], [7333.0, 0.8], [7333.0, 0.9], [7333.0, 1.0], [8358.0, 0.5], [8358.0, 0.75], [8358.0, 0.8], [8358.0, 0.9], [8358.0, 1.0], [10650.0, 0.5], [10650.0, 0.75], [10650.0, 0.8], [10650.0, 0.9], [10650.0, 1.0], [6083.0, 0.5], [6083.0, 0.75], [6083.0, 0.8], [6083.0, 0.9], [6083.0, 1.0], [1424.0, 0.5], [1424.0, 0.75], [1424.0, 0.8], [1424.0, 0.9], [1424.0, 1.0], [9281.0, 0.5], [9281.0, 0.75], [9281.0, 0.8], [9281.0, 0.9], [9281.0, 1.0], [7161.0, 0.5], [7161.0, 0.75], [7161.0, 0.8], [7161.0, 0.9], [7161.0, 1.0], [7185.0, 0.5], [7185.0, 0.75], [7185.0, 0.8], [7185.0, 0.9], [7185.0, 1.0], [12980.0, 0.5], [12980.0, 0.75], [12980.0, 0.8], [12980.0, 0.9], [12980.0, 1.0], [8282.0, 0.5], [8282.0, 0.75], [8282.0, 0.8], [8282.0, 0.9], [8282.0, 1.0], [3962.0, 0.5], [3962.0, 0.75], [3962.0, 0.8], [3962.0, 0.9], [3962.0, 1.0], [720.0, 0.5], [720.0, 0.75], [720.0, 0.8], [720.0, 0.9], [720.0, 1.0], [11723.0, 0.5], [11723.0, 0.75], [11723.0, 0.8], [11723.0, 0.9], [11723.0, 1.0], [8446.0, 0.5], [8446.0, 0.75], [8446.0, 0.8], [8446.0, 0.9], [8446.0, 1.0], [8790.0, 0.5], [8790.0, 0.75], [8790.0, 0.8], [8790.0, 0.9], [8790.0, 1.0], [1241.0, 0.5], [1241.0, 0.75], [1241.0, 0.8], [1241.0, 0.9], [1241.0, 1.0], [9098.0, 0.5], [9098.0, 0.75], [9098.0, 0.8], [9098.0, 0.9], [9098.0, 1.0], [1739.0, 0.5], [1739.0, 0.75], [1739.0, 0.8], [1739.0, 0.9], [1739.0, 1.0], [4636.0, 0.5], [4636.0, 0.75], [4636.0, 0.8], [4636.0, 0.9], [4636.0, 1.0], [8421.0, 0.5], [8421.0, 0.75], [8421.0, 0.8], [8421.0, 0.9], [8421.0, 1.0], [5215.0, 0.5], [5215.0, 0.75], [5215.0, 0.8], [5215.0, 0.9], [5215.0, 1.0], [7062.0, 0.5], [7062.0, 0.75], [7062.0, 0.8], [7062.0, 0.9], [7062.0, 1.0], [6714.0, 0.5], [6714.0, 0.75], [6714.0, 0.8], [6714.0, 0.9], [6714.0, 1.0], [7630.0, 0.5], [7630.0, 0.75], [7630.0, 0.8], [7630.0, 0.9], [7630.0, 1.0], [6970.0, 0.5], [6970.0, 0.75], [6970.0, 0.8], [6970.0, 0.9], [6970.0, 1.0], [3146.0, 0.5], [3146.0, 0.75], [3146.0, 0.8], [3146.0, 0.9], [3146.0, 1.0], [5860.0, 0.5], [5860.0, 0.75], [5860.0, 0.8], [5860.0, 0.9], [5860.0, 1.0], [7755.0, 0.5], [7755.0, 0.75], [7755.0, 0.8], [7755.0, 0.9], [7755.0, 1.0], [6105.0, 0.5], [6105.0, 0.75], [6105.0, 0.8], [6105.0, 0.9], [6105.0, 1.0], [4030.0, 0.5], [4030.0, 0.75], [4030.0, 0.8], [4030.0, 0.9], [4030.0, 1.0], [3859.0, 0.5], [3859.0, 0.75], [3859.0, 0.8], [3859.0, 0.9], [3859.0, 1.0], [4268.0, 0.5], [4268.0, 0.75], [4268.0, 0.8], [4268.0, 0.9], [4268.0, 1.0], [9409.0, 0.5], [9409.0, 0.75], [9409.0, 0.8], [9409.0, 0.9], [9409.0, 1.0], [1979.0, 0.5], [1979.0, 0.75], [1979.0, 0.8], [1979.0, 0.9], [1979.0, 1.0], [12658.0, 0.5], [12658.0, 0.75], [12658.0, 0.8], [12658.0, 0.9], [12658.0, 1.0], [5394.0, 0.5], [5394.0, 0.75], [5394.0, 0.8], [5394.0, 0.9], [5394.0, 1.0], [1210.0, 0.5], [1210.0, 0.75], [1210.0, 0.8], [1210.0, 0.9], [1210.0, 1.0], [6971.0, 0.5], [6971.0, 0.75], [6971.0, 0.8], [6971.0, 0.9], [6971.0, 1.0], [2056.0, 0.5], [2056.0, 0.75], [2056.0, 0.8], [2056.0, 0.9], [2056.0, 1.0], [2782.0, 0.5], [2782.0, 0.75], [2782.0, 0.8], [2782.0, 0.9], [2782.0, 1.0], [4974.0, 0.5], [4974.0, 0.75], [4974.0, 0.8], [4974.0, 0.9], [4974.0, 1.0], [6966.0, 0.5], [6966.0, 0.75], [6966.0, 0.8], [6966.0, 0.9], [6966.0, 1.0]], "hovertemplate": "n_orders=%{x}<br>test_roc_auc=%{y}<br>store_id=%{customdata[0]}<br>param_value=%{customdata[1]}<extra></extra>", "legendgroup": "", "marker": {"color": "#0b6fab", "symbol": "circle"}, "mode": "markers", "name": "", "orientation": "v", "showlegend": false, "type": "scatter", "x": [2570, 2570, 2570, 2570, 2570, 1999, 1999, 1999, 1999, 1999, 1077, 1077, 1077, 1077, 1077, 945, 945, 945, 945, 945, 4378, 4378, 4378, 4378, 4378, 63963, 63963, 63963, 63963, 63963, 1244, 1244, 1244, 1244, 1244, 980, 980, 980, 980, 980, 756, 756, 756, 756, 756, 3818, 3818, 3818, 3818, 3818, 11663, 11663, 11663, 11663, 11663, 664, 664, 664, 664, 664, 850, 850, 850, 850, 850, 35738, 35738, 35738, 35738, 35738, 4443, 4443, 4443, 4443, 4443, 7946, 7946, 7946, 7946, 7946, 16184, 16184, 16184, 16184, 16184, 6586, 6586, 6586, 6586, 6586, 3099, 3099, 3099, 3099, 3099, 454, 454, 454, 454, 454, 387, 387, 387, 387, 387, 1180, 1180, 1180, 1180, 1180, 14420, 14420, 14420, 14420, 14420, 698, 698, 698, 698, 698, 613, 613, 613, 613, 613, 2284, 2284, 2284, 2284, 2284, 14628, 14628, 14628, 14628, 14628, 2471, 2471, 2471, 2471, 2471, 823, 823, 823, 823, 823, 840, 840, 840, 840, 840, 4317, 4317, 4317, 4317, 4317, 10980, 10980, 10980, 10980, 10980, 3349, 3349, 3349, 3349, 3349, 1342, 1342, 1342, 1342, 1342, 41190, 41190, 41190, 41190, 41190, 3718, 3718, 3718, 3718, 3718, 999, 999, 999, 999, 999, 8867, 8867, 8867, 8867, 8867, 294, 294, 294, 294, 294, 25699, 25699, 25699, 25699, 25699, 443, 443, 443, 443, 443, 2695, 2695, 2695, 2695, 2695, 2739, 2739, 2739, 2739, 2739, 2920, 2920, 2920, 2920, 2920, 6533, 6533, 6533, 6533, 6533, 1781, 1781, 1781, 1781, 1781, 798, 798, 798, 798, 798, 5851, 5851, 5851, 5851, 5851, 3467, 3467, 3467, 3467, 3467, 867, 867, 867, 867, 867, 42877, 42877, 42877, 42877, 42877, 3287, 3287, 3287, 3287, 3287, 9202, 9202, 9202, 9202, 9202, 3534, 3534, 3534, 3534, 3534, 14109, 14109, 14109, 14109, 14109, 4269, 4269, 4269, 4269, 4269, 19152, 19152, 19152, 19152, 19152, 1115, 1115, 1115, 1115, 1115, 9931, 9931, 9931, 9931, 9931, 8434, 8434, 8434, 8434, 8434, 1994, 1994, 1994, 1994, 1994, 3533, 3533, 3533, 3533, 3533, 20577, 20577, 20577, 20577, 20577, 2489, 2489, 2489, 2489, 2489, 382, 382, 382, 382, 382, 9602, 9602, 9602, 9602, 9602, 1225, 1225, 1225, 1225, 1225, 861, 861, 861, 861, 861, 51376, 51376, 51376, 51376, 51376, 4028, 4028, 4028, 4028, 4028, 5252, 5252, 5252, 5252, 5252, 2093, 2093, 2093, 2093, 2093, 730, 730, 730, 730, 730, 206, 206, 206, 206, 206, 7501, 7501, 7501, 7501, 7501, 449, 449, 449, 449, 449, 2648, 2648, 2648, 2648, 2648, 2294, 2294, 2294, 2294, 2294, 2330, 2330, 2330, 2330, 2330, 218, 218, 218, 218, 218, 3481, 3481, 3481, 3481, 3481, 157, 157, 157, 157, 157, 1787, 1787, 1787, 1787, 1787, 1587, 1587, 1587, 1587, 1587, 225, 225, 225, 225, 225, 1127, 1127, 1127, 1127, 1127, 22839, 22839, 22839, 22839, 22839, 7451, 7451, 7451, 7451, 7451, 8006, 8006, 8006, 8006, 8006, 2812, 2812, 2812, 2812, 2812, 2564, 2564, 2564, 2564, 2564, 16512, 16512, 16512, 16512, 16512, 272, 272, 272, 272, 272, 515, 515, 515, 515, 515, 4028, 4028, 4028, 4028, 4028, 701, 701, 701, 701, 701, 28370, 28370, 28370, 28370, 28370, 8161, 8161, 8161, 8161, 8161, 215, 215, 215, 215, 215, 5893, 5893, 5893, 5893, 5893], "xaxis": "x", "y": [0.7971710449843077, 0.8012757177728699, 0.7947663605718935, 0.7780570731140299, 0.7719836103684761, 0.5545614919354839, 0.7995211693548386, 0.8469632056451614, 0.8589969758064516, 0.8397807459677419, 0.5423783287419651, 0.5636363636363636, 0.5420569329660239, 0.5356290174471994, 0.5604683195592286, 0.9182119205298014, 0.9212472406181016, 0.9161699779249448, 0.9123068432671082, 0.8881346578366445, 0.7770484671682276, 0.7293106095501306, 0.7353626081170992, 0.7551947387276728, 0.7859921183274476, 0.9373512713250606, 0.9637087888373794, 0.9696420501653488, 0.9708336929974662, 0.9722157462529042, 0.9711962833914053, 0.9602787456445993, 0.9667828106852497, 0.9714285714285715, 0.9679442508710802, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7122153209109732, 0.9296296296296296, 0.9401234567901234, 0.9384259259259259, 0.9333333333333333, 0.9265432098765433, 0.9293236344786395, 0.9421674799130033, 0.937440133424988, 0.9408321031339096, 0.934484837557331, 0.911489062342016, 0.9637564137070692, 0.9658391329052545, 0.9679752551598033, 0.9686787514223014, 0.7220052083333334, 0.8130208333333332, 0.8139322916666665, 0.815234375, 0.7630208333333333, 0.3797169811320755, 0.464622641509434, 0.4658018867924528, 0.4976415094339623, 0.49410377358490565, 0.9328715410430508, 0.9563418075366722, 0.941742560107131, 0.9631563073135956, 0.9714646418486316, 0.4948104693140794, 0.49616425992779783, 0.49526173285198555, 0.49909747292418777, 0.4749548736462094, 0.7560834955598816, 0.881755180138137, 0.8983339555721486, 0.9033374223312622, 0.8935654950798688, 0.9446806495584261, 0.938254742987414, 0.9489485771883361, 0.9470794562434254, 0.9501757163103295, 0.9662953628374056, 0.9709486312252154, 0.9702434586538168, 0.9713534525162038, 0.9727659741175549, 0.9281197742474916, 0.9252891583054625, 0.9213872630992198, 0.9220056438127091, 0.9183301978818283, 0.7600000000000001, 0.638888888888889, 0.6111111111111112, 0.6677777777777778, 0.48, 0.6431451612903226, 0.5873655913978495, 0.5782930107526881, 0.5514112903225806, 0.5413306451612904, 0.8665893679069605, 0.8710867502898704, 0.879413934858227, 0.8713678366888022, 0.8740030216787885, 0.8347402950733017, 0.8740075200827963, 0.9305247400951892, 0.9064420413937694, 0.9271555830951381, 0.6619098426246999, 0.7013870365430782, 0.7085889570552147, 0.7005868231528408, 0.702987463323553, 0.8745238095238095, 0.7783333333333333, 0.9004761904761904, 0.7854761904761903, 0.6878571428571426, 0.8324440132589538, 0.8636753173255719, 0.8849704907429865, 0.8780337941628265, 0.8745573611447975, 0.8815242081826489, 0.9016301270253921, 0.9131880542077966, 0.9177150229726773, 0.9280036614715383, 0.9289538714991762, 0.9413284409165793, 0.9264639808297139, 0.9332596974689232, 0.9260146772502621, 0.9131315987933636, 0.9103035444947211, 0.917020173453997, 0.9136500754147813, 0.9153940422322776, 0.7679738562091505, 0.7604166666666665, 0.8049428104575163, 0.764501633986928, 0.7317197712418301, 0.9568839623471452, 0.9701680302630421, 0.9682326031494677, 0.9716591888800915, 0.968223805753497, 0.8023818654523823, 0.8890566447946517, 0.8915161141290399, 0.8858619013253053, 0.8799329043479311, 0.9309480122324159, 0.9479434250764526, 0.9637308868501528, 0.9723318042813455, 0.9770107033639143, 0.9502151799687011, 0.943075117370892, 0.9305800078247262, 0.9216793818466356, 0.9462539123630673, 0.925487351374237, 0.9379563207846735, 0.9358429947429701, 0.9390341177715839, 0.9395720989309528, 0.7858541604434268, 0.8450330246606259, 0.8317484815294236, 0.8516462544392446, 0.8728467589365727, 0.7493960026356249, 0.7507870268687313, 0.7401713156160773, 0.7379749615638042, 0.7048100153744784, 0.9767149220313271, 0.9773147420853109, 0.9767895028316902, 0.9784699510781687, 0.9778225262579956, 0.777972027972028, 0.7316433566433567, 0.7447552447552448, 0.6258741258741258, 0.5305944055944056, 0.9425361927619013, 0.9426087824441853, 0.9464480088615251, 0.9481018670554868, 0.9450866033298422, 0.9616747181964573, 0.970048309178744, 0.9674718196457327, 0.9623188405797102, 0.949597423510467, 0.9045049130763417, 0.9215117157974301, 0.917838246409675, 0.9262131519274376, 0.9234467120181405, 0.8431196902271282, 0.8873509410699493, 0.8923308096861816, 0.901462174189447, 0.9035042092893333, 0.9113676398394409, 0.9347800976060527, 0.9363033873343153, 0.9325601374570447, 0.9308238759421295, 0.7907585004359198, 0.8181621960200101, 0.7878382610609356, 0.7880985432660973, 0.7916811267891757, 0.9785597572362279, 0.9837885154061625, 0.982983193277311, 0.9877684407096171, 0.9876050420168068, null, null, null, null, null, 0.8627402921953096, 0.9509227220299884, 0.9262351018838909, 0.9505094194540562, 0.9431949250288351, 0.9727111432808417, 0.9789430478073582, 0.9714797683779395, 0.9765553280135584, 0.9812336699385636, 0.9291074249605056, 0.8975118483412323, 0.8579186413902053, 0.8597946287519747, 0.6994470774091628, 0.9382198647032738, 0.9501893971244209, 0.9371661505045523, 0.948434829480631, 0.9543387563546651, 0.970843935538592, 0.9725800466497032, 0.971745122985581, 0.9697837150127226, 0.9677913839411931, 0.9688684913028447, 0.9705206390949744, 0.9704063395621856, 0.9712991275450538, 0.9735042347504584, 0.8849513688760806, 0.8804214697406341, 0.8810878962536024, 0.9125450288184438, 0.9349153458213255, 0.8170384889522452, 0.6065235210263721, 0.6225053456878118, 0.7123342836778332, 0.6768068424803992, 0.945488964419395, 0.9200189304306673, 0.9032719528460181, 0.9155767327797617, 0.905842619283225, 0.9146297287219569, 0.9358145527123277, 0.9573521716378858, 0.9631104858450875, 0.964134024275914, 0.7233754512635379, 0.9711191335740071, 0.9787906137184116, 0.9801444043321299, 0.9817238267148014, 0.949010535013427, 0.9534030159058047, 0.952968394959719, 0.9577508779177856, 0.9543334021896303, 0.9217071111882484, 0.9149915587181875, 0.9069210621814797, 0.9215611637924233, 0.8995309960712539, 0.8551335581606981, 0.8648760772977891, 0.8654381457095445, 0.8612627803650769, 0.8482281462448478, 0.6632474901789611, 0.6772846315622395, 0.8042934804174438, 0.658644498234197, 0.6121384072060633, 0.9546596500543565, 0.957997705980981, 0.9582107243826455, 0.9579004083867072, 0.9576081866179373, 0.87382079268489, 0.8765199566416605, 0.8529089796838275, 0.8679073161831783, 0.8519752728677678, 0.8910411622276029, 0.8621872477804681, 0.8405972558514931, 0.8317191283292978, 0.7610976594027442, 0.7807025710251515, 0.8110754239786497, 0.7924482924482924, 0.7807685146394824, 0.7661949758723953, 0.8948512585812357, 0.9121967963386728, 0.9162242562929063, 0.9128146453089244, 0.904096109839817, 0.9296311146752205, 0.9361467522052928, 0.9289127238706228, 0.9298483025928895, 0.9179697941726811, 0.9498580648945333, 0.9558748996625585, 0.9583512328021735, 0.9614807302025099, 0.9635393808397698, 0.8105516588733022, 0.8538187486083277, 0.8536656646626586, 0.8532203295479848, 0.83375083500334, 0.7853393271461717, 0.8867024361948954, 0.8952726218097448, 0.9261649265274554, 0.904741879350348, 0.9904397705544933, 0.9780114722753345, 0.9980879541108987, 0.9985659655831739, 0.4933078393881453, 0.8340004105090313, 0.7954125615763548, 0.814090722495895, 0.8204022988505748, 0.8185036945812808, 0.8938301282051282, 0.9002403846153846, 0.8998397435897436, 0.8810096153846154, 0.8565705128205128, 0.8898323972541677, 0.8954154408487118, 0.8893532138718017, 0.8978336453597218, 0.9008090398502274, 0.854261796042618, 0.8729071537290715, 0.873668188736682, 0.7423896499238964, 0.810882800608828, 0.5, 0.5, 0.5, 0.5, 0.5, 0.48556430446194226, 0.5253718285214348, 0.5332458442694663, 0.5328083989501313, 0.5336832895888014, 0.883217350179042, 0.9013242348489967, 0.8878060040988222, 0.8912967592279799, 0.8906211292029818, null, null, null, null, null, 0.3489138176638177, 0.5759437321937322, 0.6458110754985755, 0.6462562321937322, 0.5303151709401709, 0.8047619047619048, 0.8142857142857142, 0.8134920634920635, 0.7738095238095238, 0.6158730158730159, 0.9038074939656969, 0.9135766776707254, 0.9061001929632295, 0.9080499956297526, 0.9048698002460785, 0.6623417721518987, 0.7202531645569621, 0.7123417721518988, 0.7, 0.7564873417721518, null, null, null, null, null, 0.8609126984126984, 0.8699074074074074, 0.8657407407407407, 0.8704034391534391, 0.8617724867724869, 0.959812539374469, 0.9681700339841997, 0.9705716590570582, 0.9713039271255386, 0.972182648807715, 0.8744221315378812, 0.9022330815499696, 0.9073212545698314, 0.919102078494868, 0.886825698021144, 0.9600612114485587, 0.9644797026872498, 0.9648706824067534, 0.9663365060146416, 0.9662650366035493, 0.8721484438650806, 0.8807710064635272, 0.8809067948509044, 0.8751154201292706, 0.8691441257943621, 0.9548688516173661, 0.9687456934637928, 0.9656395831137844, 0.9650869954090635, 0.9560658997992308, 0.4173489810771471, 0.6120865680090571, 0.4625384117742196, 0.6760371179039302, 0.48982087983179695, 0.809375, 0.7875, 0.7760416666666667, 0.7921875, 0.7390625, 0.8951772737282536, 0.8389121338912133, 0.864787491741907, 0.8371504073992514, 0.8709535344637745, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6680351906158357, 0.7624633431085044, 0.7788856304985338, 0.7640762463343109, 0.3425219941348973, 0.948205937626192, 0.9545893128941073, 0.9596632031808157, 0.9611697418693504, 0.958515826421893, 0.7062305295950156, 0.679185112313494, 0.6831939662239712, 0.6847679947532382, 0.6811772421708476, 0.9122448979591836, 0.9081632653061225, 0.9081632653061225, 0.8785714285714286, 0.8760204081632653, 0.8151921253581005, 0.8163772977050985, 0.813499750221742, 0.8186915697289141, 0.8200181981302312], "yaxis": "y"}, {"hovertemplate": "<b>OLS trendline</b><br>test_roc_auc = 3.28922e-06 * n_orders + 0.818109<br>R<sup>2</sup>=0.068793<br><br>n_orders=%{x}<br>test_roc_auc=%{y} <b>(trend)</b><extra></extra>", "legendgroup": "", "marker": {"color": "#0b6fab", "symbol": "circle"}, "mode": "lines", "name": "", "showlegend": false, "type": "scatter", "x": [157, 157, 157, 157, 157, 206, 206, 206, 206, 206, 215, 215, 215, 215, 215, 272, 272, 272, 272, 272, 294, 294, 294, 294, 294, 382, 382, 382, 382, 382, 387, 387, 387, 387, 387, 443, 443, 443, 443, 443, 449, 449, 449, 449, 449, 454, 454, 454, 454, 454, 515, 515, 515, 515, 515, 613, 613, 613, 613, 613, 664, 664, 664, 664, 664, 698, 698, 698, 698, 698, 701, 701, 701, 701, 701, 730, 730, 730, 730, 730, 756, 756, 756, 756, 756, 823, 823, 823, 823, 823, 840, 840, 840, 840, 840, 850, 850, 850, 850, 850, 861, 861, 861, 861, 861, 867, 867, 867, 867, 867, 945, 945, 945, 945, 945, 980, 980, 980, 980, 980, 999, 999, 999, 999, 999, 1077, 1077, 1077, 1077, 1077, 1115, 1115, 1115, 1115, 1115, 1127, 1127, 1127, 1127, 1127, 1180, 1180, 1180, 1180, 1180, 1225, 1225, 1225, 1225, 1225, 1244, 1244, 1244, 1244, 1244, 1342, 1342, 1342, 1342, 1342, 1587, 1587, 1587, 1587, 1587, 1781, 1781, 1781, 1781, 1781, 1787, 1787, 1787, 1787, 1787, 1994, 1994, 1994, 1994, 1994, 1999, 1999, 1999, 1999, 1999, 2093, 2093, 2093, 2093, 2093, 2284, 2284, 2284, 2284, 2284, 2294, 2294, 2294, 2294, 2294, 2330, 2330, 2330, 2330, 2330, 2471, 2471, 2471, 2471, 2471, 2489, 2489, 2489, 2489, 2489, 2564, 2564, 2564, 2564, 2564, 2570, 2570, 2570, 2570, 2570, 2648, 2648, 2648, 2648, 2648, 2695, 2695, 2695, 2695, 2695, 2739, 2739, 2739, 2739, 2739, 2812, 2812, 2812, 2812, 2812, 2920, 2920, 2920, 2920, 2920, 3099, 3099, 3099, 3099, 3099, 3287, 3287, 3287, 3287, 3287, 3349, 3349, 3349, 3349, 3349, 3467, 3467, 3467, 3467, 3467, 3481, 3481, 3481, 3481, 3481, 3533, 3533, 3533, 3533, 3533, 3534, 3534, 3534, 3534, 3534, 3718, 3718, 3718, 3718, 3718, 3818, 3818, 3818, 3818, 3818, 4028, 4028, 4028, 4028, 4028, 4028, 4028, 4028, 4028, 4028, 4269, 4269, 4269, 4269, 4269, 4317, 4317, 4317, 4317, 4317, 4378, 4378, 4378, 4378, 4378, 4443, 4443, 4443, 4443, 4443, 5252, 5252, 5252, 5252, 5252, 5851, 5851, 5851, 5851, 5851, 5893, 5893, 5893, 5893, 5893, 6533, 6533, 6533, 6533, 6533, 6586, 6586, 6586, 6586, 6586, 7451, 7451, 7451, 7451, 7451, 7501, 7501, 7501, 7501, 7501, 7946, 7946, 7946, 7946, 7946, 8006, 8006, 8006, 8006, 8006, 8161, 8161, 8161, 8161, 8161, 8434, 8434, 8434, 8434, 8434, 8867, 8867, 8867, 8867, 8867, 9202, 9202, 9202, 9202, 9202, 9602, 9602, 9602, 9602, 9602, 9931, 9931, 9931, 9931, 9931, 10980, 10980, 10980, 10980, 10980, 11663, 11663, 11663, 11663, 11663, 14109, 14109, 14109, 14109, 14109, 14420, 14420, 14420, 14420, 14420, 14628, 14628, 14628, 14628, 14628, 16184, 16184, 16184, 16184, 16184, 16512, 16512, 16512, 16512, 16512, 19152, 19152, 19152, 19152, 19152, 20577, 20577, 20577, 20577, 20577, 22839, 22839, 22839, 22839, 22839, 25699, 25699, 25699, 25699, 25699, 28370, 28370, 28370, 28370, 28370, 35738, 35738, 35738, 35738, 35738, 41190, 41190, 41190, 41190, 41190, 42877, 42877, 42877, 42877, 42877, 51376, 51376, 51376, 51376, 51376, 63963, 63963, 63963, 63963, 63963], "xaxis": "x", "y": [0.8186251403044419, 0.8186251403044419, 0.8186251403044419, 0.8186251403044419, 0.8186251403044419, 0.8187863119728922, 0.8187863119728922, 0.8187863119728922, 0.8187863119728922, 0.8187863119728922, 0.8188159149324035, 0.8188159149324035, 0.8188159149324035, 0.8188159149324035, 0.8188159149324035, 0.8190034003426416, 0.8190034003426416, 0.8190034003426416, 0.8190034003426416, 0.8190034003426416, 0.819075763132558, 0.819075763132558, 0.819075763132558, 0.819075763132558, 0.819075763132558, 0.8193652142922239, 0.8193652142922239, 0.8193652142922239, 0.8193652142922239, 0.8193652142922239, 0.8193816603808413, 0.8193816603808413, 0.8193816603808413, 0.8193816603808413, 0.8193816603808413, 0.8195658565733559, 0.8195658565733559, 0.8195658565733559, 0.8195658565733559, 0.8195658565733559, 0.8195855918796968, 0.8195855918796968, 0.8195855918796968, 0.8195855918796968, 0.8195855918796968, 0.8196020379683141, 0.8196020379683141, 0.8196020379683141, 0.8196020379683141, 0.8196020379683141, 0.8198026802494461, 0.8198026802494461, 0.8198026802494461, 0.8198026802494461, 0.8198026802494461, 0.8201250235863468, 0.8201250235863468, 0.8201250235863468, 0.8201250235863468, 0.8201250235863468, 0.820292773690244, 0.820292773690244, 0.820292773690244, 0.820292773690244, 0.820292773690244, 0.8204046070928421, 0.8204046070928421, 0.8204046070928421, 0.8204046070928421, 0.8204046070928421, 0.8204144747460126, 0.8204144747460126, 0.8204144747460126, 0.8204144747460126, 0.8204144747460126, 0.8205098620599934, 0.8205098620599934, 0.8205098620599934, 0.8205098620599934, 0.8205098620599934, 0.8205953817208037, 0.8205953817208037, 0.8205953817208037, 0.8205953817208037, 0.8205953817208037, 0.8208157593082765, 0.8208157593082765, 0.8208157593082765, 0.8208157593082765, 0.8208157593082765, 0.8208716760095757, 0.8208716760095757, 0.8208716760095757, 0.8208716760095757, 0.8208716760095757, 0.8209045681868105, 0.8209045681868105, 0.8209045681868105, 0.8209045681868105, 0.8209045681868105, 0.8209407495817687, 0.8209407495817687, 0.8209407495817687, 0.8209407495817687, 0.8209407495817687, 0.8209604848881095, 0.8209604848881095, 0.8209604848881095, 0.8209604848881095, 0.8209604848881095, 0.8212170438705406, 0.8212170438705406, 0.8212170438705406, 0.8212170438705406, 0.8212170438705406, 0.8213321664908623, 0.8213321664908623, 0.8213321664908623, 0.8213321664908623, 0.8213321664908623, 0.8213946616276083, 0.8213946616276083, 0.8213946616276083, 0.8213946616276083, 0.8213946616276083, 0.8216512206100394, 0.8216512206100394, 0.8216512206100394, 0.8216512206100394, 0.8216512206100394, 0.8217762108835315, 0.8217762108835315, 0.8217762108835315, 0.8217762108835315, 0.8217762108835315, 0.8218156814962131, 0.8218156814962131, 0.8218156814962131, 0.8218156814962131, 0.8218156814962131, 0.8219900100355574, 0.8219900100355574, 0.8219900100355574, 0.8219900100355574, 0.8219900100355574, 0.8221380248331137, 0.8221380248331137, 0.8221380248331137, 0.8221380248331137, 0.8221380248331137, 0.8222005199698598, 0.8222005199698598, 0.8222005199698598, 0.8222005199698598, 0.8222005199698598, 0.8225228633067604, 0.8225228633067604, 0.8225228633067604, 0.8225228633067604, 0.8225228633067604, 0.8233287216490118, 0.8233287216490118, 0.8233287216490118, 0.8233287216490118, 0.8233287216490118, 0.8239668298873661, 0.8239668298873661, 0.8239668298873661, 0.8239668298873661, 0.8239668298873661, 0.8239865651937069, 0.8239865651937069, 0.8239865651937069, 0.8239865651937069, 0.8239865651937069, 0.8246674332624664, 0.8246674332624664, 0.8246674332624664, 0.8246674332624664, 0.8246674332624664, 0.8246838793510838, 0.8246838793510838, 0.8246838793510838, 0.8246838793510838, 0.8246838793510838, 0.8249930658170904, 0.8249930658170904, 0.8249930658170904, 0.8249930658170904, 0.8249930658170904, 0.8256213064022743, 0.8256213064022743, 0.8256213064022743, 0.8256213064022743, 0.8256213064022743, 0.8256541985795091, 0.8256541985795091, 0.8256541985795091, 0.8256541985795091, 0.8256541985795091, 0.8257726104175541, 0.8257726104175541, 0.8257726104175541, 0.8257726104175541, 0.8257726104175541, 0.8262363901165641, 0.8262363901165641, 0.8262363901165641, 0.8262363901165641, 0.8262363901165641, 0.8262955960355868, 0.8262955960355868, 0.8262955960355868, 0.8262955960355868, 0.8262955960355868, 0.8265422873648474, 0.8265422873648474, 0.8265422873648474, 0.8265422873648474, 0.8265422873648474, 0.8265620226711883, 0.8265620226711883, 0.8265620226711883, 0.8265620226711883, 0.8265620226711883, 0.8268185816536193, 0.8268185816536193, 0.8268185816536193, 0.8268185816536193, 0.8268185816536193, 0.8269731748866227, 0.8269731748866227, 0.8269731748866227, 0.8269731748866227, 0.8269731748866227, 0.8271179004664556, 0.8271179004664556, 0.8271179004664556, 0.8271179004664556, 0.8271179004664556, 0.8273580133602694, 0.8273580133602694, 0.8273580133602694, 0.8273580133602694, 0.8273580133602694, 0.8277132488744047, 0.8277132488744047, 0.8277132488744047, 0.8277132488744047, 0.8277132488744047, 0.8283020188469068, 0.8283020188469068, 0.8283020188469068, 0.8283020188469068, 0.8283020188469068, 0.8289203917789202, 0.8289203917789202, 0.8289203917789202, 0.8289203917789202, 0.8289203917789202, 0.8291243232777756, 0.8291243232777756, 0.8291243232777756, 0.8291243232777756, 0.8291243232777756, 0.8295124509691457, 0.8295124509691457, 0.8295124509691457, 0.8295124509691457, 0.8295124509691457, 0.8295585000172744, 0.8295585000172744, 0.8295585000172744, 0.8295585000172744, 0.8295585000172744, 0.8297295393388952, 0.8297295393388952, 0.8297295393388952, 0.8297295393388952, 0.8297295393388952, 0.8297328285566186, 0.8297328285566186, 0.8297328285566186, 0.8297328285566186, 0.8297328285566186, 0.8303380446177381, 0.8303380446177381, 0.8303380446177381, 0.8303380446177381, 0.8303380446177381, 0.8306669663900856, 0.8306669663900856, 0.8306669663900856, 0.8306669663900856, 0.8306669663900856, 0.8313577021120155, 0.8313577021120155, 0.8313577021120155, 0.8313577021120155, 0.8313577021120155, 0.8313577021120155, 0.8313577021120155, 0.8313577021120155, 0.8313577021120155, 0.8313577021120155, 0.8321504035833731, 0.8321504035833731, 0.8321504035833731, 0.8321504035833731, 0.8321504035833731, 0.8323082860340999, 0.8323082860340999, 0.8323082860340999, 0.8323082860340999, 0.8323082860340999, 0.8325089283152319, 0.8325089283152319, 0.8325089283152319, 0.8325089283152319, 0.8325089283152319, 0.8327227274672578, 0.8327227274672578, 0.8327227274672578, 0.8327227274672578, 0.8327227274672578, 0.8353837046055494, 0.8353837046055494, 0.8353837046055494, 0.8353837046055494, 0.8353837046055494, 0.8373539460219113, 0.8373539460219113, 0.8373539460219113, 0.8373539460219113, 0.8373539460219113, 0.8374920931662972, 0.8374920931662972, 0.8374920931662972, 0.8374920931662972, 0.8374920931662972, 0.8395971925093215, 0.8395971925093215, 0.8395971925093215, 0.8395971925093215, 0.8395971925093215, 0.8397715210486658, 0.8397715210486658, 0.8397715210486658, 0.8397715210486658, 0.8397715210486658, 0.8426166943794721, 0.8426166943794721, 0.8426166943794721, 0.8426166943794721, 0.8426166943794721, 0.8427811552656458, 0.8427811552656458, 0.8427811552656458, 0.8427811552656458, 0.8427811552656458, 0.8442448571525925, 0.8442448571525925, 0.8442448571525925, 0.8442448571525925, 0.8442448571525925, 0.844442210216001, 0.844442210216001, 0.844442210216001, 0.844442210216001, 0.844442210216001, 0.8449520389631396, 0.8449520389631396, 0.8449520389631396, 0.8449520389631396, 0.8449520389631396, 0.8458499954016485, 0.8458499954016485, 0.8458499954016485, 0.8458499954016485, 0.8458499954016485, 0.8472742266759133, 0.8472742266759133, 0.8472742266759133, 0.8472742266759133, 0.8472742266759133, 0.8483761146132777, 0.8483761146132777, 0.8483761146132777, 0.8483761146132777, 0.8483761146132777, 0.8496918017026678, 0.8496918017026678, 0.8496918017026678, 0.8496918017026678, 0.8496918017026678, 0.8507739543336913, 0.8507739543336913, 0.8507739543336913, 0.8507739543336913, 0.8507739543336913, 0.8542243437256171, 0.8542243437256171, 0.8542243437256171, 0.8542243437256171, 0.8542243437256171, 0.8564708794307507, 0.8564708794307507, 0.8564708794307507, 0.8564708794307507, 0.8564708794307507, 0.8645163059823718, 0.8645163059823718, 0.8645163059823718, 0.8645163059823718, 0.8645163059823718, 0.8655392526943727, 0.8655392526943727, 0.8655392526943727, 0.8655392526943727, 0.8655392526943727, 0.8662234099808556, 0.8662234099808556, 0.8662234099808556, 0.8662234099808556, 0.8662234099808556, 0.8713414327585834, 0.8713414327585834, 0.8713414327585834, 0.8713414327585834, 0.8713414327585834, 0.8724202961718833, 0.8724202961718833, 0.8724202961718833, 0.8724202961718833, 0.8724202961718833, 0.8811038309618586, 0.8811038309618586, 0.8811038309618586, 0.8811038309618586, 0.8811038309618586, 0.8857909662178112, 0.8857909662178112, 0.8857909662178112, 0.8857909662178112, 0.8857909662178112, 0.8932311767083128, 0.8932311767083128, 0.8932311767083128, 0.8932311767083128, 0.8932311767083128, 0.9026383393974526, 0.9026383393974526, 0.9026383393974526, 0.9026383393974526, 0.9026383393974526, 0.9114238399368556, 0.9114238399368556, 0.9114238399368556, 0.9114238399368556, 0.9114238399368556, 0.9356587961234228, 0.9356587961234228, 0.9356587961234228, 0.9356587961234228, 0.9356587961234228, 0.9535916111518112, 0.9535916111518112, 0.9535916111518112, 0.9535916111518112, 0.9535916111518112, 0.9591405214513143, 0.9591405214513143, 0.9591405214513143, 0.9591405214513143, 0.9591405214513143, 0.9870955828831324, 0.9870955828831324, 0.9870955828831324, 0.9870955828831324, 0.9870955828831324, 1.0284969663685182, 1.0284969663685182, 1.0284969663685182, 1.0284969663685182, 1.0284969663685182], "yaxis": "y"}],                        {"height": 500, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Relationship between test_roc_auc and dataset sizes"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "n_orders"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "test_roc_auc"}}},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('11236549-ae15-4db9-bd07-568c7993f704');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>


```python
# Select a performance metric:
metric = 'test_roc_auc'

px.scatter(data_frame=metrics_subsample,
           x='n_vars', y=metric, hover_data=['store_id', 'param_value'],
           color_discrete_sequence=['#0b6fab'],
           width=900, height=500, title='Relationship between ' + metric + ' and number of features')
```


<div>                            <div id="31d9b741-212a-48ab-9f75-9452a598eb36" class="plotly-graph-div" style="height:500px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("31d9b741-212a-48ab-9f75-9452a598eb36")) {                    Plotly.newPlot(                        "31d9b741-212a-48ab-9f75-9452a598eb36",                        [{"customdata": [[11729.0, 0.5], [11729.0, 0.75], [11729.0, 0.8], [11729.0, 0.9], [11729.0, 1.0], [10311.0, 0.5], [10311.0, 0.75], [10311.0, 0.8], [10311.0, 0.9], [10311.0, 1.0], [7988.0, 0.5], [7988.0, 0.75], [7988.0, 0.8], [7988.0, 0.9], [7988.0, 1.0], [4736.0, 0.5], [4736.0, 0.75], [4736.0, 0.8], [4736.0, 0.9], [4736.0, 1.0], [3481.0, 0.5], [3481.0, 0.75], [3481.0, 0.8], [3481.0, 0.9], [3481.0, 1.0], [4838.0, 0.5], [4838.0, 0.75], [4838.0, 0.8], [4838.0, 0.9], [4838.0, 1.0], [5848.0, 0.5], [5848.0, 0.75], [5848.0, 0.8], [5848.0, 0.9], [5848.0, 1.0], [6106.0, 0.5], [6106.0, 0.75], [6106.0, 0.8], [6106.0, 0.9], [6106.0, 1.0], [1559.0, 0.5], [1559.0, 0.75], [1559.0, 0.8], [1559.0, 0.9], [1559.0, 1.0], [5342.0, 0.5], [5342.0, 0.75], [5342.0, 0.8], [5342.0, 0.9], [5342.0, 1.0], [3781.0, 0.5], [3781.0, 0.75], [3781.0, 0.8], [3781.0, 0.9], [3781.0, 1.0], [4408.0, 0.5], [4408.0, 0.75], [4408.0, 0.8], [4408.0, 0.9], [4408.0, 1.0], [7292.0, 0.5], [7292.0, 0.75], [7292.0, 0.8], [7292.0, 0.9], [7292.0, 1.0], [6044.0, 0.5], [6044.0, 0.75], [6044.0, 0.8], [6044.0, 0.9], [6044.0, 1.0], [8181.0, 0.5], [8181.0, 0.75], [8181.0, 0.8], [8181.0, 0.9], [8181.0, 1.0], [2352.0, 0.5], [2352.0, 0.75], [2352.0, 0.8], [2352.0, 0.9], [2352.0, 1.0], [9491.0, 0.5], [9491.0, 0.75], [9491.0, 0.8], [9491.0, 0.9], [9491.0, 1.0], [5847.0, 0.5], [5847.0, 0.75], [5847.0, 0.8], [5847.0, 0.9], [5847.0, 1.0], [7939.0, 0.5], [7939.0, 0.75], [7939.0, 0.8], [7939.0, 0.9], [7939.0, 1.0], [6078.0, 0.5], [6078.0, 0.75], [6078.0, 0.8], [6078.0, 0.9], [6078.0, 1.0], [10268.0, 0.5], [10268.0, 0.75], [10268.0, 0.8], [10268.0, 0.9], [10268.0, 1.0], [10060.0, 0.5], [10060.0, 0.75], [10060.0, 0.8], [10060.0, 0.9], [10060.0, 1.0], [6256.0, 0.5], [6256.0, 0.75], [6256.0, 0.8], [6256.0, 0.9], [6256.0, 1.0], [8436.0, 0.5], [8436.0, 0.75], [8436.0, 0.8], [8436.0, 0.9], [8436.0, 1.0], [5085.0, 0.5], [5085.0, 0.75], [5085.0, 0.8], [5085.0, 0.9], [5085.0, 1.0], [8783.0, 0.5], [8783.0, 0.75], [8783.0, 0.8], [8783.0, 0.9], [8783.0, 1.0], [6047.0, 0.5], [6047.0, 0.75], [6047.0, 0.8], [6047.0, 0.9], [6047.0, 1.0], [8832.0, 0.5], [8832.0, 0.75], [8832.0, 0.8], [8832.0, 0.9], [8832.0, 1.0], [1961.0, 0.5], [1961.0, 0.75], [1961.0, 0.8], [1961.0, 0.9], [1961.0, 1.0], [7845.0, 0.5], [7845.0, 0.75], [7845.0, 0.8], [7845.0, 0.9], [7845.0, 1.0], [2699.0, 0.5], [2699.0, 0.75], [2699.0, 0.8], [2699.0, 0.9], [2699.0, 1.0], [6004.0, 0.5], [6004.0, 0.75], [6004.0, 0.8], [6004.0, 0.9], [6004.0, 1.0], [2868.0, 0.5], [2868.0, 0.75], [2868.0, 0.8], [2868.0, 0.9], [2868.0, 1.0], [1875.0, 0.5], [1875.0, 0.75], [1875.0, 0.8], [1875.0, 0.9], [1875.0, 1.0], [5593.0, 0.5], [5593.0, 0.75], [5593.0, 0.8], [5593.0, 0.9], [5593.0, 1.0], [7849.0, 0.5], [7849.0, 0.75], [7849.0, 0.8], [7849.0, 0.9], [7849.0, 1.0], [11223.0, 0.5], [11223.0, 0.75], [11223.0, 0.8], [11223.0, 0.9], [11223.0, 1.0], [6170.0, 0.5], [6170.0, 0.75], [6170.0, 0.8], [6170.0, 0.9], [6170.0, 1.0], [5168.0, 0.5], [5168.0, 0.75], [5168.0, 0.8], [5168.0, 0.9], [5168.0, 1.0], [2866.0, 0.5], [2866.0, 0.75], [2866.0, 0.8], [2866.0, 0.9], [2866.0, 1.0], [3437.0, 0.5], [3437.0, 0.75], [3437.0, 0.8], [3437.0, 0.9], [3437.0, 1.0], [6929.0, 0.5], [6929.0, 0.75], [6929.0, 0.8], [6929.0, 0.9], [6929.0, 1.0], [8894.0, 0.5], [8894.0, 0.75], [8894.0, 0.8], [8894.0, 0.9], [8894.0, 1.0], [9177.0, 0.5], [9177.0, 0.75], [9177.0, 0.8], [9177.0, 0.9], [9177.0, 1.0], [10349.0, 0.5], [10349.0, 0.75], [10349.0, 0.8], [10349.0, 0.9], [10349.0, 1.0], [3988.0, 0.5], [3988.0, 0.75], [3988.0, 0.8], [3988.0, 0.9], [3988.0, 1.0], [1549.0, 0.5], [1549.0, 0.75], [1549.0, 0.8], [1549.0, 0.9], [1549.0, 1.0], [9541.0, 0.5], [9541.0, 0.75], [9541.0, 0.8], [9541.0, 0.9], [9541.0, 1.0], [1181.0, 0.5], [1181.0, 0.75], [1181.0, 0.8], [1181.0, 0.9], [1181.0, 1.0], [7790.0, 0.5], [7790.0, 0.75], [7790.0, 0.8], [7790.0, 0.9], [7790.0, 1.0], [5663.0, 0.5], [5663.0, 0.75], [5663.0, 0.8], [5663.0, 0.9], [5663.0, 1.0], [4601.0, 0.5], [4601.0, 0.75], [4601.0, 0.8], [4601.0, 0.9], [4601.0, 1.0], [1603.0, 0.5], [1603.0, 0.75], [1603.0, 0.8], [1603.0, 0.9], [1603.0, 1.0], [2212.0, 0.5], [2212.0, 0.75], [2212.0, 0.8], [2212.0, 0.9], [2212.0, 1.0], [9761.0, 0.5], [9761.0, 0.75], [9761.0, 0.8], [9761.0, 0.9], [9761.0, 1.0], [5428.0, 0.5], [5428.0, 0.75], [5428.0, 0.8], [5428.0, 0.9], [5428.0, 1.0], [1098.0, 0.5], [1098.0, 0.75], [1098.0, 0.8], [1098.0, 0.9], [1098.0, 1.0], [5939.0, 0.5], [5939.0, 0.75], [5939.0, 0.8], [5939.0, 0.9], [5939.0, 1.0], [7333.0, 0.5], [7333.0, 0.75], [7333.0, 0.8], [7333.0, 0.9], [7333.0, 1.0], [8358.0, 0.5], [8358.0, 0.75], [8358.0, 0.8], [8358.0, 0.9], [8358.0, 1.0], [10650.0, 0.5], [10650.0, 0.75], [10650.0, 0.8], [10650.0, 0.9], [10650.0, 1.0], [6083.0, 0.5], [6083.0, 0.75], [6083.0, 0.8], [6083.0, 0.9], [6083.0, 1.0], [1424.0, 0.5], [1424.0, 0.75], [1424.0, 0.8], [1424.0, 0.9], [1424.0, 1.0], [9281.0, 0.5], [9281.0, 0.75], [9281.0, 0.8], [9281.0, 0.9], [9281.0, 1.0], [7161.0, 0.5], [7161.0, 0.75], [7161.0, 0.8], [7161.0, 0.9], [7161.0, 1.0], [7185.0, 0.5], [7185.0, 0.75], [7185.0, 0.8], [7185.0, 0.9], [7185.0, 1.0], [12980.0, 0.5], [12980.0, 0.75], [12980.0, 0.8], [12980.0, 0.9], [12980.0, 1.0], [8282.0, 0.5], [8282.0, 0.75], [8282.0, 0.8], [8282.0, 0.9], [8282.0, 1.0], [3962.0, 0.5], [3962.0, 0.75], [3962.0, 0.8], [3962.0, 0.9], [3962.0, 1.0], [720.0, 0.5], [720.0, 0.75], [720.0, 0.8], [720.0, 0.9], [720.0, 1.0], [11723.0, 0.5], [11723.0, 0.75], [11723.0, 0.8], [11723.0, 0.9], [11723.0, 1.0], [8446.0, 0.5], [8446.0, 0.75], [8446.0, 0.8], [8446.0, 0.9], [8446.0, 1.0], [8790.0, 0.5], [8790.0, 0.75], [8790.0, 0.8], [8790.0, 0.9], [8790.0, 1.0], [1241.0, 0.5], [1241.0, 0.75], [1241.0, 0.8], [1241.0, 0.9], [1241.0, 1.0], [9098.0, 0.5], [9098.0, 0.75], [9098.0, 0.8], [9098.0, 0.9], [9098.0, 1.0], [1739.0, 0.5], [1739.0, 0.75], [1739.0, 0.8], [1739.0, 0.9], [1739.0, 1.0], [4636.0, 0.5], [4636.0, 0.75], [4636.0, 0.8], [4636.0, 0.9], [4636.0, 1.0], [8421.0, 0.5], [8421.0, 0.75], [8421.0, 0.8], [8421.0, 0.9], [8421.0, 1.0], [5215.0, 0.5], [5215.0, 0.75], [5215.0, 0.8], [5215.0, 0.9], [5215.0, 1.0], [7062.0, 0.5], [7062.0, 0.75], [7062.0, 0.8], [7062.0, 0.9], [7062.0, 1.0], [6714.0, 0.5], [6714.0, 0.75], [6714.0, 0.8], [6714.0, 0.9], [6714.0, 1.0], [7630.0, 0.5], [7630.0, 0.75], [7630.0, 0.8], [7630.0, 0.9], [7630.0, 1.0], [6970.0, 0.5], [6970.0, 0.75], [6970.0, 0.8], [6970.0, 0.9], [6970.0, 1.0], [3146.0, 0.5], [3146.0, 0.75], [3146.0, 0.8], [3146.0, 0.9], [3146.0, 1.0], [5860.0, 0.5], [5860.0, 0.75], [5860.0, 0.8], [5860.0, 0.9], [5860.0, 1.0], [7755.0, 0.5], [7755.0, 0.75], [7755.0, 0.8], [7755.0, 0.9], [7755.0, 1.0], [6105.0, 0.5], [6105.0, 0.75], [6105.0, 0.8], [6105.0, 0.9], [6105.0, 1.0], [4030.0, 0.5], [4030.0, 0.75], [4030.0, 0.8], [4030.0, 0.9], [4030.0, 1.0], [3859.0, 0.5], [3859.0, 0.75], [3859.0, 0.8], [3859.0, 0.9], [3859.0, 1.0], [4268.0, 0.5], [4268.0, 0.75], [4268.0, 0.8], [4268.0, 0.9], [4268.0, 1.0], [9409.0, 0.5], [9409.0, 0.75], [9409.0, 0.8], [9409.0, 0.9], [9409.0, 1.0], [1979.0, 0.5], [1979.0, 0.75], [1979.0, 0.8], [1979.0, 0.9], [1979.0, 1.0], [12658.0, 0.5], [12658.0, 0.75], [12658.0, 0.8], [12658.0, 0.9], [12658.0, 1.0], [5394.0, 0.5], [5394.0, 0.75], [5394.0, 0.8], [5394.0, 0.9], [5394.0, 1.0], [1210.0, 0.5], [1210.0, 0.75], [1210.0, 0.8], [1210.0, 0.9], [1210.0, 1.0], [6971.0, 0.5], [6971.0, 0.75], [6971.0, 0.8], [6971.0, 0.9], [6971.0, 1.0], [2056.0, 0.5], [2056.0, 0.75], [2056.0, 0.8], [2056.0, 0.9], [2056.0, 1.0], [2782.0, 0.5], [2782.0, 0.75], [2782.0, 0.8], [2782.0, 0.9], [2782.0, 1.0], [4974.0, 0.5], [4974.0, 0.75], [4974.0, 0.8], [4974.0, 0.9], [4974.0, 1.0], [6966.0, 0.5], [6966.0, 0.75], [6966.0, 0.8], [6966.0, 0.9], [6966.0, 1.0]], "hovertemplate": "n_vars=%{x}<br>test_roc_auc=%{y}<br>store_id=%{customdata[0]}<br>param_value=%{customdata[1]}<extra></extra>", "legendgroup": "", "marker": {"color": "#0b6fab", "symbol": "circle"}, "mode": "markers", "name": "", "orientation": "v", "showlegend": false, "type": "scatter", "x": [1596, 1596, 1596, 1596, 1596, 2534, 2534, 2534, 2534, 2534, 3051, 3051, 3051, 3051, 3051, 2393, 2393, 2393, 2393, 2393, 2545, 2545, 2545, 2545, 2545, 2017, 2017, 2017, 2017, 2017, 2986, 2986, 2986, 2986, 2986, 2386, 2386, 2386, 2386, 2386, 2590, 2590, 2590, 2590, 2590, 2367, 2367, 2367, 2367, 2367, 2649, 2649, 2649, 2649, 2649, 2231, 2231, 2231, 2231, 2231, 2030, 2030, 2030, 2030, 2030, 2271, 2271, 2271, 2271, 2271, 2150, 2150, 2150, 2150, 2150, 2469, 2469, 2469, 2469, 2469, 1628, 1628, 1628, 1628, 1628, 3883, 3883, 3883, 3883, 3883, 3248, 3248, 3248, 3248, 3248, 3608, 3608, 3608, 3608, 3608, 1991, 1991, 1991, 1991, 1991, 2395, 2395, 2395, 2395, 2395, 2359, 2359, 2359, 2359, 2359, 1985, 1985, 1985, 1985, 1985, 2144, 2144, 2144, 2144, 2144, 1608, 1608, 1608, 1608, 1608, 2053, 2053, 2053, 2053, 2053, 2769, 2769, 2769, 2769, 2769, 2381, 2381, 2381, 2381, 2381, 2001, 2001, 2001, 2001, 2001, 2321, 2321, 2321, 2321, 2321, 2069, 2069, 2069, 2069, 2069, 2402, 2402, 2402, 2402, 2402, 2476, 2476, 2476, 2476, 2476, 2074, 2074, 2074, 2074, 2074, 2030, 2030, 2030, 2030, 2030, 1416, 1416, 1416, 1416, 1416, 2391, 2391, 2391, 2391, 2391, 2279, 2279, 2279, 2279, 2279, 3457, 3457, 3457, 3457, 3457, 3557, 3557, 3557, 3557, 3557, 2058, 2058, 2058, 2058, 2058, 2039, 2039, 2039, 2039, 2039, 2282, 2282, 2282, 2282, 2282, 2340, 2340, 2340, 2340, 2340, 2315, 2315, 2315, 2315, 2315, 2140, 2140, 2140, 2140, 2140, 2071, 2071, 2071, 2071, 2071, 2698, 2698, 2698, 2698, 2698, 2220, 2220, 2220, 2220, 2220, 2079, 2079, 2079, 2079, 2079, 2559, 2559, 2559, 2559, 2559, 2324, 2324, 2324, 2324, 2324, 2527, 2527, 2527, 2527, 2527, 2696, 2696, 2696, 2696, 2696, 2076, 2076, 2076, 2076, 2076, 4026, 4026, 4026, 4026, 4026, 2463, 2463, 2463, 2463, 2463, 2549, 2549, 2549, 2549, 2549, 2479, 2479, 2479, 2479, 2479, 1606, 1606, 1606, 1606, 1606, 2035, 2035, 2035, 2035, 2035, 2726, 2726, 2726, 2726, 2726, 1602, 1602, 1602, 1602, 1602, 2263, 2263, 2263, 2263, 2263, 2197, 2197, 2197, 2197, 2197, 1643, 1643, 1643, 1643, 1643, 2073, 2073, 2073, 2073, 2073, 2536, 2536, 2536, 2536, 2536, 1858, 1858, 1858, 1858, 1858, 2067, 2067, 2067, 2067, 2067, 2673, 2673, 2673, 2673, 2673, 2058, 2058, 2058, 2058, 2058, 3791, 3791, 3791, 3791, 3791, 2642, 2642, 2642, 2642, 2642, 2476, 2476, 2476, 2476, 2476, 2454, 2454, 2454, 2454, 2454, 2074, 2074, 2074, 2074, 2074, 2292, 2292, 2292, 2292, 2292, 1772, 1772, 1772, 1772, 1772, 2327, 2327, 2327, 2327, 2327, 2294, 2294, 2294, 2294, 2294, 2465, 2465, 2465, 2465, 2465, 2494, 2494, 2494, 2494, 2494, 2816, 2816, 2816, 2816, 2816, 1981, 1981, 1981, 1981, 1981, 2283, 2283, 2283, 2283, 2283, 2163, 2163, 2163, 2163, 2163, 1837, 1837, 1837, 1837, 1837, 2483, 2483, 2483, 2483, 2483, 2235, 2235, 2235, 2235, 2235, 2687, 2687, 2687, 2687, 2687, 1676, 1676, 1676, 1676, 1676, 2676, 2676, 2676, 2676, 2676, 2101, 2101, 2101, 2101, 2101, 2863, 2863, 2863, 2863, 2863, 2516, 2516, 2516, 2516, 2516, 2495, 2495, 2495, 2495, 2495, 2115, 2115, 2115, 2115, 2115, 2049, 2049, 2049, 2049, 2049], "xaxis": "x", "y": [0.7971710449843077, 0.8012757177728699, 0.7947663605718935, 0.7780570731140299, 0.7719836103684761, 0.5545614919354839, 0.7995211693548386, 0.8469632056451614, 0.8589969758064516, 0.8397807459677419, 0.5423783287419651, 0.5636363636363636, 0.5420569329660239, 0.5356290174471994, 0.5604683195592286, 0.9182119205298014, 0.9212472406181016, 0.9161699779249448, 0.9123068432671082, 0.8881346578366445, 0.7770484671682276, 0.7293106095501306, 0.7353626081170992, 0.7551947387276728, 0.7859921183274476, 0.9373512713250606, 0.9637087888373794, 0.9696420501653488, 0.9708336929974662, 0.9722157462529042, 0.9711962833914053, 0.9602787456445993, 0.9667828106852497, 0.9714285714285715, 0.9679442508710802, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7122153209109732, 0.9296296296296296, 0.9401234567901234, 0.9384259259259259, 0.9333333333333333, 0.9265432098765433, 0.9293236344786395, 0.9421674799130033, 0.937440133424988, 0.9408321031339096, 0.934484837557331, 0.911489062342016, 0.9637564137070692, 0.9658391329052545, 0.9679752551598033, 0.9686787514223014, 0.7220052083333334, 0.8130208333333332, 0.8139322916666665, 0.815234375, 0.7630208333333333, 0.3797169811320755, 0.464622641509434, 0.4658018867924528, 0.4976415094339623, 0.49410377358490565, 0.9328715410430508, 0.9563418075366722, 0.941742560107131, 0.9631563073135956, 0.9714646418486316, 0.4948104693140794, 0.49616425992779783, 0.49526173285198555, 0.49909747292418777, 0.4749548736462094, 0.7560834955598816, 0.881755180138137, 0.8983339555721486, 0.9033374223312622, 0.8935654950798688, 0.9446806495584261, 0.938254742987414, 0.9489485771883361, 0.9470794562434254, 0.9501757163103295, 0.9662953628374056, 0.9709486312252154, 0.9702434586538168, 0.9713534525162038, 0.9727659741175549, 0.9281197742474916, 0.9252891583054625, 0.9213872630992198, 0.9220056438127091, 0.9183301978818283, 0.7600000000000001, 0.638888888888889, 0.6111111111111112, 0.6677777777777778, 0.48, 0.6431451612903226, 0.5873655913978495, 0.5782930107526881, 0.5514112903225806, 0.5413306451612904, 0.8665893679069605, 0.8710867502898704, 0.879413934858227, 0.8713678366888022, 0.8740030216787885, 0.8347402950733017, 0.8740075200827963, 0.9305247400951892, 0.9064420413937694, 0.9271555830951381, 0.6619098426246999, 0.7013870365430782, 0.7085889570552147, 0.7005868231528408, 0.702987463323553, 0.8745238095238095, 0.7783333333333333, 0.9004761904761904, 0.7854761904761903, 0.6878571428571426, 0.8324440132589538, 0.8636753173255719, 0.8849704907429865, 0.8780337941628265, 0.8745573611447975, 0.8815242081826489, 0.9016301270253921, 0.9131880542077966, 0.9177150229726773, 0.9280036614715383, 0.9289538714991762, 0.9413284409165793, 0.9264639808297139, 0.9332596974689232, 0.9260146772502621, 0.9131315987933636, 0.9103035444947211, 0.917020173453997, 0.9136500754147813, 0.9153940422322776, 0.7679738562091505, 0.7604166666666665, 0.8049428104575163, 0.764501633986928, 0.7317197712418301, 0.9568839623471452, 0.9701680302630421, 0.9682326031494677, 0.9716591888800915, 0.968223805753497, 0.8023818654523823, 0.8890566447946517, 0.8915161141290399, 0.8858619013253053, 0.8799329043479311, 0.9309480122324159, 0.9479434250764526, 0.9637308868501528, 0.9723318042813455, 0.9770107033639143, 0.9502151799687011, 0.943075117370892, 0.9305800078247262, 0.9216793818466356, 0.9462539123630673, 0.925487351374237, 0.9379563207846735, 0.9358429947429701, 0.9390341177715839, 0.9395720989309528, 0.7858541604434268, 0.8450330246606259, 0.8317484815294236, 0.8516462544392446, 0.8728467589365727, 0.7493960026356249, 0.7507870268687313, 0.7401713156160773, 0.7379749615638042, 0.7048100153744784, 0.9767149220313271, 0.9773147420853109, 0.9767895028316902, 0.9784699510781687, 0.9778225262579956, 0.777972027972028, 0.7316433566433567, 0.7447552447552448, 0.6258741258741258, 0.5305944055944056, 0.9425361927619013, 0.9426087824441853, 0.9464480088615251, 0.9481018670554868, 0.9450866033298422, 0.9616747181964573, 0.970048309178744, 0.9674718196457327, 0.9623188405797102, 0.949597423510467, 0.9045049130763417, 0.9215117157974301, 0.917838246409675, 0.9262131519274376, 0.9234467120181405, 0.8431196902271282, 0.8873509410699493, 0.8923308096861816, 0.901462174189447, 0.9035042092893333, 0.9113676398394409, 0.9347800976060527, 0.9363033873343153, 0.9325601374570447, 0.9308238759421295, 0.7907585004359198, 0.8181621960200101, 0.7878382610609356, 0.7880985432660973, 0.7916811267891757, 0.9785597572362279, 0.9837885154061625, 0.982983193277311, 0.9877684407096171, 0.9876050420168068, null, null, null, null, null, 0.8627402921953096, 0.9509227220299884, 0.9262351018838909, 0.9505094194540562, 0.9431949250288351, 0.9727111432808417, 0.9789430478073582, 0.9714797683779395, 0.9765553280135584, 0.9812336699385636, 0.9291074249605056, 0.8975118483412323, 0.8579186413902053, 0.8597946287519747, 0.6994470774091628, 0.9382198647032738, 0.9501893971244209, 0.9371661505045523, 0.948434829480631, 0.9543387563546651, 0.970843935538592, 0.9725800466497032, 0.971745122985581, 0.9697837150127226, 0.9677913839411931, 0.9688684913028447, 0.9705206390949744, 0.9704063395621856, 0.9712991275450538, 0.9735042347504584, 0.8849513688760806, 0.8804214697406341, 0.8810878962536024, 0.9125450288184438, 0.9349153458213255, 0.8170384889522452, 0.6065235210263721, 0.6225053456878118, 0.7123342836778332, 0.6768068424803992, 0.945488964419395, 0.9200189304306673, 0.9032719528460181, 0.9155767327797617, 0.905842619283225, 0.9146297287219569, 0.9358145527123277, 0.9573521716378858, 0.9631104858450875, 0.964134024275914, 0.7233754512635379, 0.9711191335740071, 0.9787906137184116, 0.9801444043321299, 0.9817238267148014, 0.949010535013427, 0.9534030159058047, 0.952968394959719, 0.9577508779177856, 0.9543334021896303, 0.9217071111882484, 0.9149915587181875, 0.9069210621814797, 0.9215611637924233, 0.8995309960712539, 0.8551335581606981, 0.8648760772977891, 0.8654381457095445, 0.8612627803650769, 0.8482281462448478, 0.6632474901789611, 0.6772846315622395, 0.8042934804174438, 0.658644498234197, 0.6121384072060633, 0.9546596500543565, 0.957997705980981, 0.9582107243826455, 0.9579004083867072, 0.9576081866179373, 0.87382079268489, 0.8765199566416605, 0.8529089796838275, 0.8679073161831783, 0.8519752728677678, 0.8910411622276029, 0.8621872477804681, 0.8405972558514931, 0.8317191283292978, 0.7610976594027442, 0.7807025710251515, 0.8110754239786497, 0.7924482924482924, 0.7807685146394824, 0.7661949758723953, 0.8948512585812357, 0.9121967963386728, 0.9162242562929063, 0.9128146453089244, 0.904096109839817, 0.9296311146752205, 0.9361467522052928, 0.9289127238706228, 0.9298483025928895, 0.9179697941726811, 0.9498580648945333, 0.9558748996625585, 0.9583512328021735, 0.9614807302025099, 0.9635393808397698, 0.8105516588733022, 0.8538187486083277, 0.8536656646626586, 0.8532203295479848, 0.83375083500334, 0.7853393271461717, 0.8867024361948954, 0.8952726218097448, 0.9261649265274554, 0.904741879350348, 0.9904397705544933, 0.9780114722753345, 0.9980879541108987, 0.9985659655831739, 0.4933078393881453, 0.8340004105090313, 0.7954125615763548, 0.814090722495895, 0.8204022988505748, 0.8185036945812808, 0.8938301282051282, 0.9002403846153846, 0.8998397435897436, 0.8810096153846154, 0.8565705128205128, 0.8898323972541677, 0.8954154408487118, 0.8893532138718017, 0.8978336453597218, 0.9008090398502274, 0.854261796042618, 0.8729071537290715, 0.873668188736682, 0.7423896499238964, 0.810882800608828, 0.5, 0.5, 0.5, 0.5, 0.5, 0.48556430446194226, 0.5253718285214348, 0.5332458442694663, 0.5328083989501313, 0.5336832895888014, 0.883217350179042, 0.9013242348489967, 0.8878060040988222, 0.8912967592279799, 0.8906211292029818, null, null, null, null, null, 0.3489138176638177, 0.5759437321937322, 0.6458110754985755, 0.6462562321937322, 0.5303151709401709, 0.8047619047619048, 0.8142857142857142, 0.8134920634920635, 0.7738095238095238, 0.6158730158730159, 0.9038074939656969, 0.9135766776707254, 0.9061001929632295, 0.9080499956297526, 0.9048698002460785, 0.6623417721518987, 0.7202531645569621, 0.7123417721518988, 0.7, 0.7564873417721518, null, null, null, null, null, 0.8609126984126984, 0.8699074074074074, 0.8657407407407407, 0.8704034391534391, 0.8617724867724869, 0.959812539374469, 0.9681700339841997, 0.9705716590570582, 0.9713039271255386, 0.972182648807715, 0.8744221315378812, 0.9022330815499696, 0.9073212545698314, 0.919102078494868, 0.886825698021144, 0.9600612114485587, 0.9644797026872498, 0.9648706824067534, 0.9663365060146416, 0.9662650366035493, 0.8721484438650806, 0.8807710064635272, 0.8809067948509044, 0.8751154201292706, 0.8691441257943621, 0.9548688516173661, 0.9687456934637928, 0.9656395831137844, 0.9650869954090635, 0.9560658997992308, 0.4173489810771471, 0.6120865680090571, 0.4625384117742196, 0.6760371179039302, 0.48982087983179695, 0.809375, 0.7875, 0.7760416666666667, 0.7921875, 0.7390625, 0.8951772737282536, 0.8389121338912133, 0.864787491741907, 0.8371504073992514, 0.8709535344637745, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6680351906158357, 0.7624633431085044, 0.7788856304985338, 0.7640762463343109, 0.3425219941348973, 0.948205937626192, 0.9545893128941073, 0.9596632031808157, 0.9611697418693504, 0.958515826421893, 0.7062305295950156, 0.679185112313494, 0.6831939662239712, 0.6847679947532382, 0.6811772421708476, 0.9122448979591836, 0.9081632653061225, 0.9081632653061225, 0.8785714285714286, 0.8760204081632653, 0.8151921253581005, 0.8163772977050985, 0.813499750221742, 0.8186915697289141, 0.8200181981302312], "yaxis": "y"}],                        {"height": 500, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Relationship between test_roc_auc and number of features"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "n_vars"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "test_roc_auc"}}},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('31d9b741-212a-48ab-9f75-9452a598eb36');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


<a id='reference4'></a>
#### Distribution of performance metric by best hyper-parameter value


```python
px.strip(data_frame=best_subsample_values.sort_values('param_value'),
         x=best_subsample_values.sort_values('param_value')['param_value'].apply(lambda x: 'eta = ' + str(x)),
         y=best_subsample_values.sort_values('param_value')['test_roc_auc'],
         hover_data=['store_id', 'param_value'],
         color_discrete_sequence=['#0b6fab'],
         width=900, height=500, title='Distribution of test_roc_auc by best subsample value',
         labels={'y': 'test_roc_auc', 'x': ''})
```


<div>                            <div id="78372e3c-dec9-45a9-9ad3-ab54a427c515" class="plotly-graph-div" style="height:500px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("78372e3c-dec9-45a9-9ad3-ab54a427c515")) {                    Plotly.newPlot(                        "78372e3c-dec9-45a9-9ad3-ab54a427c515",                        [{"alignmentgroup": "True", "boxpoints": "all", "customdata": [[6078.0, 0.5], [5168.0, 0.5], [5394.0, 0.5], [5428.0, 0.5], [12658.0, 0.5], [6106.0, 0.5], [7161.0, 0.5], [7790.0, 0.5], [7939.0, 0.5], [2782.0, 0.5], [4636.0, 0.5], [8358.0, 0.5], [4974.0, 0.5], [1875.0, 0.5], [1210.0, 0.5], [9761.0, 0.5], [8790.0, 0.5], [10268.0, 0.5], [8832.0, 0.75], [11729.0, 0.75], [5215.0, 0.75], [5342.0, 0.75], [11223.0, 0.75], [10349.0, 0.75], [6970.0, 0.75], [7630.0, 0.75], [9541.0, 0.75], [9409.0, 0.75], [9281.0, 0.75], [8282.0, 0.75], [7185.0, 0.75], [7988.0, 0.75], [720.0, 0.75], [4736.0, 0.75], [4601.0, 0.75], [1241.0, 0.75], [1559.0, 0.75], [3437.0, 0.75], [10650.0, 0.8], [1424.0, 0.8], [10060.0, 0.8], [1739.0, 0.8], [1961.0, 0.8], [9177.0, 0.8], [8783.0, 0.8], [8436.0, 0.8], [7845.0, 0.8], [6971.0, 0.8], [6256.0, 0.8], [12980.0, 0.8], [6083.0, 0.8], [5085.0, 0.8], [6004.0, 0.8], [4268.0, 0.8], [7755.0, 0.9], [1979.0, 0.9], [2056.0, 0.9], [5848.0, 0.9], [8446.0, 0.9], [2352.0, 0.9], [8181.0, 0.9], [2699.0, 0.9], [2866.0, 0.9], [4030.0, 0.9], [3988.0, 0.9], [7292.0, 0.9], [10311.0, 0.9], [4408.0, 0.9], [11723.0, 0.9], [6929.0, 0.9], [6714.0, 0.9], [6170.0, 0.9], [3859.0, 0.9], [7333.0, 0.9], [5663.0, 1.0], [1603.0, 1.0], [1181.0, 1.0], [9491.0, 1.0], [5593.0, 1.0], [1098.0, 1.0], [5847.0, 1.0], [6105.0, 1.0], [8894.0, 1.0], [5939.0, 1.0], [2212.0, 1.0], [7849.0, 1.0], [2868.0, 1.0], [4838.0, 1.0], [3146.0, 1.0], [6044.0, 1.0], [3481.0, 1.0], [6047.0, 1.0], [3781.0, 1.0], [6966.0, 1.0], [3962.0, 1.0], [9098.0, 1.0], [8421.0, 1.0]], "fillcolor": "rgba(255,255,255,0)", "hoveron": "points", "hovertemplate": "=%{x}<br>test_roc_auc=%{y}<br>store_id=%{customdata[0]}<br>param_value=%{customdata[1]}<extra></extra>", "legendgroup": "", "line": {"color": "rgba(255,255,255,0)"}, "marker": {"color": "#0b6fab"}, "name": "", "offsetgroup": "", "orientation": "v", "pointpos": 0, "showlegend": false, "type": "box", "x": ["eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.5", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.75", "eta = 0.8", "eta = 0.8", "eta = 0.8", "eta = 0.8", "eta = 0.8", "eta = 0.8", "eta = 0.8", "eta = 0.8", "eta = 0.8", "eta = 0.8", "eta = 0.8", "eta = 0.8", "eta = 0.8", "eta = 0.8", "eta = 0.8", "eta = 0.8", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 0.9", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0", "eta = 1.0"], "x0": " ", "xaxis": "x", "y": [0.7600000000000001, 0.777972027972028, 0.8951772737282536, 0.945488964419395, 0.809375, 0.7142857142857143, 0.8910411622276029, 0.9291074249605056, 0.9281197742474916, 0.7062305295950156, 0.5, 0.9217071111882484, 0.9122448979591836, 0.9502151799687011, 0.5, 0.8170384889522452, 0.8340004105090313, 0.6431451612903226, 0.9413284409165793, 0.8012757177728699, 0.9013242348489967, 0.9421674799130033, 0.7507870268687313, 0.8181621960200101, 0.9135766776707254, 0.8142857142857142, 0.9509227220299884, 0.9687456934637928, 0.8765199566416605, 0.9361467522052928, 0.8110754239786497, 0.5636363636363636, 0.8538187486083277, 0.9212472406181016, 0.9725800466497032, 0.9002403846153846, 0.9401234567901234, 0.970048309178744, 0.8654381457095445, 0.9582107243826455, 0.879413934858227, 0.873668188736682, 0.917020173453997, 0.9363033873343153, 0.8849704907429865, 0.7085889570552147, 0.8049428104575163, 0.7788856304985338, 0.9305247400951892, 0.9162242562929063, 0.8042934804174438, 0.9004761904761904, 0.8915161141290399, 0.8809067948509044, 0.8704034391534391, 0.6760371179039302, 0.9611697418693504, 0.9714285714285715, 0.9985659655831739, 0.9033374223312622, 0.49909747292418777, 0.9716591888800915, 0.9481018670554868, 0.919102078494868, 0.9877684407096171, 0.4976415094339623, 0.8589969758064516, 0.815234375, 0.9261649265274554, 0.9262131519274376, 0.6462562321937322, 0.9784699510781687, 0.9663365060146416, 0.9577508779177856, 0.9543387563546651, 0.9735042347504584, 0.9812336699385636, 0.9501757163103295, 0.9395720989309528, 0.964134024275914, 0.9727659741175549, 0.972182648807715, 0.9035042092893333, 0.9817238267148014, 0.9349153458213255, 0.8728467589365727, 0.9770107033639143, 0.9722157462529042, 0.7564873417721518, 0.9714646418486316, 0.7859921183274476, 0.9280036614715383, 0.9686787514223014, 0.8200181981302312, 0.9635393808397698, 0.9008090398502274, 0.5336832895888014], "y0": " ", "yaxis": "y"}],                        {"boxmode": "group", "height": 500, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Distribution of test_roc_auc by best subsample value"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": ""}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "test_roc_auc"}}},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('78372e3c-dec9-45a9-9ad3-ab54a427c515');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

<a id='reference8'></a>
#### Correlation between performance metric of best hyper-parameter value and dataset information


```python
# Generate a mask for the upper triangle:
mask = np.triu(np.ones_like(best_subsample_values[['test_roc_auc', 'n_orders', 'n_vars', 'avg_y']].corr(),
                            dtype=np.bool))

sns.heatmap(best_subsample_values[['test_roc_auc', 'n_orders', 'n_vars', 'avg_y']].corr(),
            mask = mask, annot = True, cmap = 'viridis')
plt.title('Correlation between performance metric and dataset information')
plt.tight_layout()
```


![png](output_99_0.png)



```python
# Generate masks for the upper triangle:
mask05 = np.triu(np.ones_like(metrics_subsample[metrics_subsample.param_value==0.5][['test_roc_auc', 
                                                                                 'n_orders', 'n_vars',
                                                                                 'avg_y']].corr(), dtype=np.bool))
mask075 = np.triu(np.ones_like(metrics_subsample[metrics_subsample.param_value==0.75][['test_roc_auc', 
                                                                                 'n_orders', 'n_vars',
                                                                                 'avg_y']].corr(), dtype=np.bool))
mask1 = np.triu(np.ones_like(metrics_subsample[metrics_subsample.param_value==1][['test_roc_auc', 
                                                                                 'n_orders', 'n_vars',
                                                                                 'avg_y']].corr(), dtype=np.bool))
```


```python
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))

corr_matrices = sns.heatmap(metrics_subsample[metrics_subsample.param_value==0.5][['test_roc_auc', 'n_orders',
                                                                 'n_vars', 'avg_y']].corr(),
                            mask = mask05, annot = True, cmap = 'viridis', ax=axs[0])
sns.heatmap(metrics_subsample[metrics_subsample.param_value==0.75][['test_roc_auc', 'n_orders',
                                                                 'n_vars', 'avg_y']].corr(),
            mask = mask075, annot = True, cmap = 'viridis', ax=axs[1])
sns.heatmap(metrics_subsample[metrics_subsample.param_value==1][['test_roc_auc', 'n_orders',
                                                                 'n_vars', 'avg_y']].corr(),
            mask = mask1, annot = True, cmap = 'viridis', ax=axs[2])


axs[0].set_title('subsample = 0.5', loc='left')
axs[1].set_title('subsample = 0.75', loc='left')
axs[2].set_title('subsample = 1', loc='left')

plt.tight_layout()
```


![png](output_101_0.png)


[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

<a id='max_depth'></a>

## Max depth

Since outcomes for $n\_estimators = 500$ produced similar conclusions as those for $n\_estimators = 1000$, only results for the first specification are presented and discussed below.

Main findings are listed below:
* Ensembles with small trees performed better, on average, than those with large maximum depth.
* Maximum depth relates inversely with dataset length.

<a id='proc_data_max_depth'></a>

### Processing data

#### Performance metrics


```python
# Assessing missing hyper-parameter values:
for s in tun_max_depth['500'].keys():
    for m in ['test_roc_auc', 'test_prec_avg', 'test_pr_auc', 'test_deviance', 'test_brier_score']:
        if len(tun_max_depth['500'][s][m].keys()) != 10:
            print('Missing hyper-parameter value for store ' + str(s) + ' and metric ' + m + '!')
```


```python
# Collecting reference data:
param_value = []
stores = []

# Loop over datasets:
for s in tun_max_depth['500'].keys():
    # Loop over hyper-parameter values:
    for v in tun_max_depth['500'][s]['test_roc_auc'].keys():
        param_value.append(float(v))
        stores.append(int(s))

metrics_max_depth = pd.DataFrame(data=param_value, columns=['param_value'], index=stores)

# Collecting performance metrics:
for m in ['test_roc_auc', 'test_prec_avg', 'test_pr_auc', 'test_deviance', 'test_brier_score']:
    stores = []
    param_value = []
    ref = []
    
    # Loop over datasets:
    for s in tun_max_depth['500'].keys():
        # Loop over hyper-parameter values:
        for v in tun_max_depth['500'][s][m].keys():
            stores.append(int(s))
            ref.append(float(tun_max_depth['500'][s][m][v]))

    metrics_max_depth = pd.concat([metrics_max_depth, pd.DataFrame(data={m: ref},
                                                                   index=stores)], axis=1)

metrics_max_depth.index.name = 'store_id'
metrics_max_depth.reset_index(inplace=True, drop=False)
print('\033[1mShape of metrics_max_depth:\033[0m ' + str(metrics_max_depth.shape) + '.')
metrics_max_depth.index.name = 'n_est_500'
metrics_max_depth.head()
```

    [1mShape of metrics_max_depth:[0m (1000, 7).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>param_value</th>
      <th>test_roc_auc</th>
      <th>test_prec_avg</th>
      <th>test_pr_auc</th>
      <th>test_deviance</th>
      <th>test_brier_score</th>
    </tr>
    <tr>
      <th>n_est_500</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11729</td>
      <td>1.0</td>
      <td>0.799801</td>
      <td>0.202104</td>
      <td>0.190858</td>
      <td>883.126107</td>
      <td>0.042397</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11729</td>
      <td>2.0</td>
      <td>0.792224</td>
      <td>0.168588</td>
      <td>0.162599</td>
      <td>886.386542</td>
      <td>0.044716</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11729</td>
      <td>3.0</td>
      <td>0.767131</td>
      <td>0.193678</td>
      <td>0.188664</td>
      <td>886.381210</td>
      <td>0.042318</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11729</td>
      <td>4.0</td>
      <td>0.767399</td>
      <td>0.155830</td>
      <td>0.150081</td>
      <td>886.556028</td>
      <td>0.050345</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11729</td>
      <td>5.0</td>
      <td>0.762692</td>
      <td>0.163812</td>
      <td>0.157680</td>
      <td>887.063994</td>
      <td>0.044718</td>
    </tr>
  </tbody>
</table>
</div>



<a id='stats_max_depth'></a>

### Statistics by hyper-parameter value

#### Basic statistics for each performance metric


```python
# Test ROC-AUC:
metrics_max_depth.groupby('param_value').describe()[['test_roc_auc']].sort_values(('test_roc_auc','mean'),
                                                                                     ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_roc_auc</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.0</th>
      <td>97.0</td>
      <td>0.841187</td>
      <td>0.148197</td>
      <td>0.425634</td>
      <td>0.799801</td>
      <td>0.894203</td>
      <td>0.945233</td>
      <td>0.983908</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>97.0</td>
      <td>0.825810</td>
      <td>0.150996</td>
      <td>0.465879</td>
      <td>0.729167</td>
      <td>0.878254</td>
      <td>0.944719</td>
      <td>0.987558</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>97.0</td>
      <td>0.825704</td>
      <td>0.151371</td>
      <td>0.342522</td>
      <td>0.756858</td>
      <td>0.876992</td>
      <td>0.941152</td>
      <td>0.989621</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>97.0</td>
      <td>0.815601</td>
      <td>0.156948</td>
      <td>0.342522</td>
      <td>0.729167</td>
      <td>0.883343</td>
      <td>0.938803</td>
      <td>0.976326</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>97.0</td>
      <td>0.804321</td>
      <td>0.163414</td>
      <td>0.342522</td>
      <td>0.707682</td>
      <td>0.866204</td>
      <td>0.929771</td>
      <td>0.976286</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>97.0</td>
      <td>0.801520</td>
      <td>0.163155</td>
      <td>0.342522</td>
      <td>0.707006</td>
      <td>0.869632</td>
      <td>0.929517</td>
      <td>0.977266</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>97.0</td>
      <td>0.799921</td>
      <td>0.159337</td>
      <td>0.340212</td>
      <td>0.698893</td>
      <td>0.858816</td>
      <td>0.929307</td>
      <td>0.979190</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>97.0</td>
      <td>0.796034</td>
      <td>0.160304</td>
      <td>0.342522</td>
      <td>0.698893</td>
      <td>0.860084</td>
      <td>0.931680</td>
      <td>0.971219</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>97.0</td>
      <td>0.790829</td>
      <td>0.161324</td>
      <td>0.342097</td>
      <td>0.683002</td>
      <td>0.836210</td>
      <td>0.925192</td>
      <td>0.971589</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>97.0</td>
      <td>0.789882</td>
      <td>0.160268</td>
      <td>0.345015</td>
      <td>0.675557</td>
      <td>0.843485</td>
      <td>0.923188</td>
      <td>0.969461</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test average precision score:
metrics_max_depth.groupby('param_value').describe()[['test_prec_avg']].sort_values(('test_prec_avg','mean'),
                                                                                      ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_prec_avg</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.0</th>
      <td>97.0</td>
      <td>0.426941</td>
      <td>0.273931</td>
      <td>0.000955</td>
      <td>0.187745</td>
      <td>0.433220</td>
      <td>0.637111</td>
      <td>0.958824</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>97.0</td>
      <td>0.402451</td>
      <td>0.277224</td>
      <td>0.000957</td>
      <td>0.152394</td>
      <td>0.401729</td>
      <td>0.625110</td>
      <td>0.951153</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>97.0</td>
      <td>0.397657</td>
      <td>0.270624</td>
      <td>0.000955</td>
      <td>0.153609</td>
      <td>0.393805</td>
      <td>0.571968</td>
      <td>0.963264</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>97.0</td>
      <td>0.382716</td>
      <td>0.269435</td>
      <td>0.000955</td>
      <td>0.155757</td>
      <td>0.379758</td>
      <td>0.547551</td>
      <td>0.946950</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>97.0</td>
      <td>0.371813</td>
      <td>0.267998</td>
      <td>0.000955</td>
      <td>0.123284</td>
      <td>0.364669</td>
      <td>0.545421</td>
      <td>0.945034</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>97.0</td>
      <td>0.371563</td>
      <td>0.262966</td>
      <td>0.000955</td>
      <td>0.137383</td>
      <td>0.366164</td>
      <td>0.558613</td>
      <td>0.944771</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>97.0</td>
      <td>0.368960</td>
      <td>0.264474</td>
      <td>0.000955</td>
      <td>0.149057</td>
      <td>0.361240</td>
      <td>0.566827</td>
      <td>0.939298</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>97.0</td>
      <td>0.368943</td>
      <td>0.259287</td>
      <td>0.000955</td>
      <td>0.141054</td>
      <td>0.358118</td>
      <td>0.565056</td>
      <td>0.936805</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>97.0</td>
      <td>0.364097</td>
      <td>0.258891</td>
      <td>0.000955</td>
      <td>0.149441</td>
      <td>0.353629</td>
      <td>0.561312</td>
      <td>0.937545</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>97.0</td>
      <td>0.361578</td>
      <td>0.259191</td>
      <td>0.000955</td>
      <td>0.132341</td>
      <td>0.361462</td>
      <td>0.542218</td>
      <td>0.935539</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test Brier score:
metrics_max_depth.groupby('param_value').describe()[['test_brier_score']].sort_values(('test_brier_score',
                                                                                          'mean'),
                                                                                          ascending=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_brier_score</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.0</th>
      <td>97.0</td>
      <td>0.032823</td>
      <td>0.035779</td>
      <td>0.001986</td>
      <td>0.010401</td>
      <td>0.015987</td>
      <td>0.039450</td>
      <td>0.194459</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>97.0</td>
      <td>0.035347</td>
      <td>0.042071</td>
      <td>0.001986</td>
      <td>0.010675</td>
      <td>0.017662</td>
      <td>0.044560</td>
      <td>0.291535</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>97.0</td>
      <td>0.037713</td>
      <td>0.045067</td>
      <td>0.001986</td>
      <td>0.012145</td>
      <td>0.017770</td>
      <td>0.042318</td>
      <td>0.259594</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>97.0</td>
      <td>0.039558</td>
      <td>0.047411</td>
      <td>0.001696</td>
      <td>0.012281</td>
      <td>0.019784</td>
      <td>0.049748</td>
      <td>0.303914</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>97.0</td>
      <td>0.040711</td>
      <td>0.048724</td>
      <td>0.001986</td>
      <td>0.012520</td>
      <td>0.020009</td>
      <td>0.049730</td>
      <td>0.310055</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>97.0</td>
      <td>0.040738</td>
      <td>0.050700</td>
      <td>0.001986</td>
      <td>0.012128</td>
      <td>0.020872</td>
      <td>0.050131</td>
      <td>0.334709</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>97.0</td>
      <td>0.041567</td>
      <td>0.053825</td>
      <td>0.001986</td>
      <td>0.011976</td>
      <td>0.019547</td>
      <td>0.052518</td>
      <td>0.318168</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>97.0</td>
      <td>0.043040</td>
      <td>0.051975</td>
      <td>0.001986</td>
      <td>0.012071</td>
      <td>0.021142</td>
      <td>0.049568</td>
      <td>0.307962</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>97.0</td>
      <td>0.043933</td>
      <td>0.056046</td>
      <td>0.001986</td>
      <td>0.012262</td>
      <td>0.022174</td>
      <td>0.048213</td>
      <td>0.325822</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>97.0</td>
      <td>0.044396</td>
      <td>0.057021</td>
      <td>0.001986</td>
      <td>0.012949</td>
      <td>0.021921</td>
      <td>0.049198</td>
      <td>0.312928</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test binomial deviance:
metrics_max_depth.groupby('param_value').describe()[['test_deviance']].sort_values(('test_deviance','mean'),
                                                                                      ascending=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_deviance</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2.0</th>
      <td>97.0</td>
      <td>2426.147665</td>
      <td>3871.125218</td>
      <td>53.324620</td>
      <td>344.767153</td>
      <td>948.088369</td>
      <td>2672.433325</td>
      <td>22105.023547</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>97.0</td>
      <td>2426.386405</td>
      <td>3871.859300</td>
      <td>53.256342</td>
      <td>342.130251</td>
      <td>948.710862</td>
      <td>2675.599571</td>
      <td>22106.268100</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>97.0</td>
      <td>2427.221373</td>
      <td>3871.897265</td>
      <td>53.591019</td>
      <td>345.214135</td>
      <td>949.504788</td>
      <td>2700.972259</td>
      <td>22108.358513</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>97.0</td>
      <td>2429.756062</td>
      <td>3872.463923</td>
      <td>53.578234</td>
      <td>345.201517</td>
      <td>949.867042</td>
      <td>2704.052297</td>
      <td>22109.552005</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>97.0</td>
      <td>2430.503570</td>
      <td>3874.713018</td>
      <td>53.212640</td>
      <td>344.720362</td>
      <td>949.363402</td>
      <td>2722.227362</td>
      <td>22113.826921</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>97.0</td>
      <td>2430.829649</td>
      <td>3872.111048</td>
      <td>53.593222</td>
      <td>345.674943</td>
      <td>950.243216</td>
      <td>2718.823836</td>
      <td>22108.703532</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>97.0</td>
      <td>2430.868484</td>
      <td>3873.741551</td>
      <td>53.420216</td>
      <td>345.509969</td>
      <td>949.255666</td>
      <td>2723.489066</td>
      <td>22113.388842</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>97.0</td>
      <td>2431.887961</td>
      <td>3874.615304</td>
      <td>53.433251</td>
      <td>345.586796</td>
      <td>949.593043</td>
      <td>2723.438027</td>
      <td>22115.963760</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>97.0</td>
      <td>2432.671022</td>
      <td>3874.449826</td>
      <td>53.493874</td>
      <td>344.172118</td>
      <td>949.885844</td>
      <td>2721.690689</td>
      <td>22114.963005</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>97.0</td>
      <td>2432.774813</td>
      <td>3873.958678</td>
      <td>53.534943</td>
      <td>344.385685</td>
      <td>951.416242</td>
      <td>2723.058690</td>
      <td>22111.666717</td>
    </tr>
  </tbody>
</table>
</div>



<a id='reference10'></a>
#### Averages of performance metrics by hyper-parameter value


```python
metrics_max_depth.groupby('param_value').mean().sort_values('test_roc_auc', ascending=False).drop('store_id',
                                                                                                     axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test_roc_auc</th>
      <th>test_prec_avg</th>
      <th>test_pr_auc</th>
      <th>test_deviance</th>
      <th>test_brier_score</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.0</th>
      <td>0.841187</td>
      <td>0.426941</td>
      <td>0.439019</td>
      <td>2426.386405</td>
      <td>0.032823</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>0.825810</td>
      <td>0.402451</td>
      <td>0.414524</td>
      <td>2426.147665</td>
      <td>0.035347</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>0.825704</td>
      <td>0.397657</td>
      <td>0.413852</td>
      <td>2427.221373</td>
      <td>0.037713</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>0.815601</td>
      <td>0.382716</td>
      <td>0.399175</td>
      <td>2429.756062</td>
      <td>0.039558</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>0.804321</td>
      <td>0.371813</td>
      <td>0.393864</td>
      <td>2430.829649</td>
      <td>0.041567</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>0.801520</td>
      <td>0.368960</td>
      <td>0.395578</td>
      <td>2430.868484</td>
      <td>0.040711</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>0.799921</td>
      <td>0.371563</td>
      <td>0.395777</td>
      <td>2430.503570</td>
      <td>0.040738</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>0.796034</td>
      <td>0.364097</td>
      <td>0.397419</td>
      <td>2432.774813</td>
      <td>0.044396</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>0.790829</td>
      <td>0.361578</td>
      <td>0.388535</td>
      <td>2432.671022</td>
      <td>0.043933</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>0.789882</td>
      <td>0.368943</td>
      <td>0.394987</td>
      <td>2431.887961</td>
      <td>0.043040</td>
    </tr>
  </tbody>
</table>
</div>



[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

#### Frequency of best hyper-parameter values


```python
best_max_dep_values = metrics_max_depth.groupby('store_id').idxmax()['test_roc_auc'].values
print('\033[1mRelative frequency of highest performance metric by hyper-parameter value:\033[0m')
print(metrics_max_depth.reindex(best_max_dep_values).param_value.value_counts()/len(best_max_dep_values))
```

    [1mRelative frequency of highest performance metric by hyper-parameter value:[0m
    1.0     0.43
    2.0     0.21
    3.0     0.10
    5.0     0.05
    4.0     0.05
    6.0     0.04
    8.0     0.03
    9.0     0.02
    7.0     0.02
    10.0    0.02
    Name: param_value, dtype: float64
    

#### Average performance metric by best hyper-parameter value


```python
# Dataframe with best hyper-parameter value by dataset:
best_max_dep_values = metrics_max_depth.reindex(best_max_dep_values)[['store_id', 'param_value',
                                                                           'test_roc_auc']]
best_max_dep_values = best_max_dep_values.merge(data_info, on='store_id', how='inner')
print('\033[1mShape of best_max_dep_values:\033[0m ' + str(best_max_dep_values.shape) + '.')
best_max_dep_values.head()
```

    [1mShape of best_max_dep_values:[0m (97, 6).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>param_value</th>
      <th>test_roc_auc</th>
      <th>n_orders</th>
      <th>n_vars</th>
      <th>avg_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>720.0</td>
      <td>2.0</td>
      <td>0.835490</td>
      <td>4028</td>
      <td>1858</td>
      <td>0.011668</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1098.0</td>
      <td>1.0</td>
      <td>0.968112</td>
      <td>19152</td>
      <td>4026</td>
      <td>0.023705</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1181.0</td>
      <td>1.0</td>
      <td>0.983908</td>
      <td>3467</td>
      <td>2698</td>
      <td>0.033458</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1210.0</td>
      <td>1.0</td>
      <td>0.500000</td>
      <td>4028</td>
      <td>2101</td>
      <td>0.001490</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1241.0</td>
      <td>1.0</td>
      <td>0.887821</td>
      <td>206</td>
      <td>3791</td>
      <td>0.320388</td>
    </tr>
  </tbody>
</table>
</div>




```python
best_max_dep_values.groupby('param_value').mean().sort_values('test_roc_auc', ascending=False)[['test_roc_auc']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test_roc_auc</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6.0</th>
      <td>0.908504</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>0.902775</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>0.894004</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>0.882947</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.874373</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>0.853647</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>0.819399</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>0.813259</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>0.762852</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>0.732427</td>
    </tr>
  </tbody>
</table>
</div>



<a id='describing_max_depth_values'></a>

### Describing hyper-parameter values

#### Average numbers of observations and features by best hyper-parameter value


```python
best_max_dep_values.groupby('param_value').mean()[['n_orders']].sort_values('n_orders', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_orders</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5.0</th>
      <td>26288.800000</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>15446.400000</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>6814.000000</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>6788.238095</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>5827.604651</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>4200.400000</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>3685.500000</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>2874.333333</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>2116.500000</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>1332.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
best_max_dep_values.groupby('param_value').mean()[['n_vars']].sort_values('n_vars', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_vars</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10.0</th>
      <td>2772.500000</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>2456.300000</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>2391.500000</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>2358.400000</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>2357.465116</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>2353.000000</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>2330.333333</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>2327.952381</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>2290.600000</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>2261.500000</td>
    </tr>
  </tbody>
</table>
</div>



#### Average of response variable by best hyper-parameter value


```python
best_max_dep_values.groupby('param_value').mean()[['avg_y']].sort_values('avg_y', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_y</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8.0</th>
      <td>0.080393</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.061088</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>0.057387</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>0.044043</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>0.043365</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>0.042684</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>0.035540</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>0.032908</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>0.007898</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>0.006682</td>
    </tr>
  </tbody>
</table>
</div>



#### Most frequent best hyper-parameter values by quartile of number of observations


```python
best_max_dep_values['quartile_n_orders'] = percentile_cut(best_max_dep_values.n_orders, p=4)['percentile']

print('\033[1mFrequency of best hyper-parameter values by quartile of number of observations:\033[0m')
for q in range(1,5):
    print('\033[1mNumber of orders in ' +
          str(np.sort(np.unique(percentile_cut(best_max_dep_values.n_orders, p=4)['interval']))[q-1]) +
          ' (quartile ' + str(q) + ')\033[0m:')
    print(best_max_dep_values[best_max_dep_values.quartile_n_orders==q].param_value.value_counts())
    print('\n')
```

    [1mFrequency of best hyper-parameter values by quartile of number of observations:[0m
    [1mNumber of orders in (156.998, 999.0] (quartile 1)[0m:
    1.0    14
    2.0     5
    3.0     3
    8.0     1
    9.0     1
    6.0     1
    Name: param_value, dtype: int64
    
    
    [1mNumber of orders in (999.0, 2812.0] (quartile 2)[0m:
    1.0     10
    3.0      4
    2.0      4
    10.0     2
    4.0      1
    5.0      1
    7.0      1
    6.0      1
    Name: param_value, dtype: int64
    
    
    [1mNumber of orders in (2812.0, 7946.0] (quartile 3)[0m:
    1.0    12
    2.0     4
    8.0     2
    3.0     2
    6.0     2
    5.0     1
    9.0     1
    Name: param_value, dtype: int64
    
    
    [1mNumber of orders in (7946.0, 63963.001] (quartile 4)[0m:
    2.0    8
    1.0    7
    4.0    4
    5.0    3
    7.0    1
    3.0    1
    Name: param_value, dtype: int64
    
    
    

#### Most frequent best hyper-parameter values by quartile of number of features


```python
best_max_dep_values['quartile_n_vars'] = percentile_cut(best_max_dep_values.n_vars, p=4)['percentile']

print('\033[1mFrequency of best hyper-parameter values by quartile of number of features:\033[0m')
for q in range(1,5):
    print('\033[1mNumber of vars in ' +
          str(np.sort(np.unique(percentile_cut(best_max_dep_values.n_vars, p=4)['interval']))[q-1]) +
          ' (quartile ' + str(q) + ')\033[0m:')
    print(best_max_dep_values[best_max_dep_values.quartile_n_vars==q].param_value.value_counts())
    print('\n')
```

    [1mFrequency of best hyper-parameter values by quartile of number of features:[0m
    [1mNumber of vars in (1415.998, 2069.0] (quartile 1)[0m:
    1.0    14
    2.0     4
    8.0     2
    4.0     1
    3.0     1
    9.0     1
    7.0     1
    5.0     1
    Name: param_value, dtype: int64
    
    
    [1mNumber of vars in (2069.0, 2321.0] (quartile 2)[0m:
    1.0    10
    2.0     7
    3.0     4
    5.0     1
    4.0     1
    6.0     1
    Name: param_value, dtype: int64
    
    
    [1mNumber of vars in (2321.0, 2534.0] (quartile 3)[0m:
    1.0     8
    2.0     6
    6.0     3
    3.0     2
    4.0     2
    7.0     1
    10.0    1
    5.0     1
    Name: param_value, dtype: int64
    
    
    [1mNumber of vars in (2534.0, 4026.001] (quartile 4)[0m:
    1.0     11
    2.0      4
    3.0      3
    5.0      2
    4.0      1
    10.0     1
    8.0      1
    9.0      1
    Name: param_value, dtype: int64
    
    
    

#### Correlation between performance metric of best hyper-parameter value and dataset information


```python
best_max_dep_values[['test_roc_auc', 'n_orders', 'n_vars', 'avg_y']].corr()[['test_roc_auc']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test_roc_auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>test_roc_auc</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>n_orders</th>
      <td>0.282984</td>
    </tr>
    <tr>
      <th>n_vars</th>
      <td>0.077852</td>
    </tr>
    <tr>
      <th>avg_y</th>
      <td>0.149174</td>
    </tr>
  </tbody>
</table>
</div>



#### Correlation between performance metric and dataset information by hyper-parameter value


```python
metrics_max_depth = metrics_max_depth.merge(data_info, on='store_id', how='left')
print('\033[1mShape of metrics_max_depth:\033[0m ' + str(metrics_max_depth.shape) + '.')
metrics_max_depth.head()
```

    [1mShape of metrics_max_depth:[0m (1000, 10).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>param_value</th>
      <th>test_roc_auc</th>
      <th>test_prec_avg</th>
      <th>test_pr_auc</th>
      <th>test_deviance</th>
      <th>test_brier_score</th>
      <th>n_orders</th>
      <th>n_vars</th>
      <th>avg_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11729</td>
      <td>1.0</td>
      <td>0.799801</td>
      <td>0.202104</td>
      <td>0.190858</td>
      <td>883.126107</td>
      <td>0.042397</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11729</td>
      <td>2.0</td>
      <td>0.792224</td>
      <td>0.168588</td>
      <td>0.162599</td>
      <td>886.386542</td>
      <td>0.044716</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11729</td>
      <td>3.0</td>
      <td>0.767131</td>
      <td>0.193678</td>
      <td>0.188664</td>
      <td>886.381210</td>
      <td>0.042318</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11729</td>
      <td>4.0</td>
      <td>0.767399</td>
      <td>0.155830</td>
      <td>0.150081</td>
      <td>886.556028</td>
      <td>0.050345</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11729</td>
      <td>5.0</td>
      <td>0.762692</td>
      <td>0.163812</td>
      <td>0.157680</td>
      <td>887.063994</td>
      <td>0.044718</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
  </tbody>
</table>
</div>




```python
for v in metrics_max_depth.param_value.unique():
    print('\033[1mmax_depth = ' + str(v) + '\033[0m')
    print(metrics_max_depth[metrics_max_depth.param_value==v][['test_roc_auc',
                                                               'n_orders', 'n_vars',
                                                               'avg_y']].corr()[['test_roc_auc']])
    print('\n')
```

    [1mmax_depth = 1.0[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.238079
    n_vars            0.053221
    avg_y             0.185016
    
    
    [1mmax_depth = 2.0[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.284880
    n_vars            0.065660
    avg_y             0.197535
    
    
    [1mmax_depth = 3.0[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.309374
    n_vars            0.074121
    avg_y             0.171135
    
    
    [1mmax_depth = 4.0[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.348119
    n_vars            0.062314
    avg_y             0.153233
    
    
    [1mmax_depth = 5.0[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.328632
    n_vars            0.021363
    avg_y             0.158820
    
    
    [1mmax_depth = 6.0[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.321188
    n_vars            0.011467
    avg_y             0.165897
    
    
    [1mmax_depth = 7.0[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.341669
    n_vars            0.005625
    avg_y             0.151090
    
    
    [1mmax_depth = 8.0[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.351387
    n_vars            0.023040
    avg_y             0.152939
    
    
    [1mmax_depth = 9.0[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.367462
    n_vars            0.018386
    avg_y             0.153566
    
    
    [1mmax_depth = 10.0[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.347318
    n_vars            0.024389
    avg_y             0.152751
    
    
    

<a id='data_vis_max_depth'></a>

### Data visualization

<a id='reference11'></a>
#### Average of performance metric by hyper-parameter value


```python
# Select a performance metric:
metric = 'test_roc_auc'

fig=px.scatter(x=metrics_max_depth['param_value'].unique(),
               y=metrics_max_depth.groupby('param_value').mean()[metric], 
               error_y=np.array(metrics_max_depth.groupby('param_value').std()['test_roc_auc']),
               color_discrete_sequence=['#0b6fab'],
               width=900, height=500,
               title='Average of ' + metric + ' by max depth value',
               labels={'y': metric, 'x': 'max_depth'})

fig.add_trace(
    go.Scatter(
        x=metrics_max_depth['param_value'].unique(),
        y=metrics_max_depth.groupby('param_value').mean()[metric],
        line = dict(color='#0b6fab', width=2, dash='dash'),
        name='avg_' + metric
              )
)

```


<div>                            <div id="f50a779f-0ded-4077-b7fb-c52144ffc795" class="plotly-graph-div" style="height:500px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("f50a779f-0ded-4077-b7fb-c52144ffc795")) {                    Plotly.newPlot(                        "f50a779f-0ded-4077-b7fb-c52144ffc795",                        [{"error_y": {"array": [0.14819651049524923, 0.1509964625169578, 0.1513714918595032, 0.15694813814180789, 0.16341378107963614, 0.1631545086034803, 0.15933690140937992, 0.16132429382735483, 0.1602681901857639, 0.1603040081518129]}, "hovertemplate": "max_depth=%{x}<br>test_roc_auc=%{y}<extra></extra>", "legendgroup": "", "marker": {"color": "#0b6fab", "symbol": "circle"}, "mode": "markers", "name": "", "orientation": "v", "showlegend": false, "type": "scatter", "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], "xaxis": "x", "y": [0.8411866069700323, 0.8258095425703209, 0.8257042595978739, 0.8156006038202637, 0.8043209443803843, 0.801520114047169, 0.799921054613861, 0.7908293122641981, 0.7898817948033606, 0.7960335592022939], "yaxis": "y"}, {"line": {"color": "#0b6fab", "dash": "dash", "width": 2}, "name": "avg_test_roc_auc", "type": "scatter", "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], "y": [0.8411866069700323, 0.8258095425703209, 0.8257042595978739, 0.8156006038202637, 0.8043209443803843, 0.801520114047169, 0.799921054613861, 0.7908293122641981, 0.7898817948033606, 0.7960335592022939]}],                        {"height": 500, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Average of test_roc_auc by max depth value"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "max_depth"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "test_roc_auc"}}},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('f50a779f-0ded-4077-b7fb-c52144ffc795');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


#### Boxplot of performance metric by hyper-parameter value


```python
# Select a performance metric:
metric = 'test_roc_auc'

px.box(data_frame=metrics_max_depth,
       x=metrics_max_depth['param_value'],
       y=metric, hover_data=['store_id'],
       color_discrete_sequence=['#0b6fab'],
       width=900, height=500,
       labels={'param_value': 'max_deph'},
       title='Distribution of ' + metric + ' by max depth value')
```


<div>                            <div id="aa8f989d-1d5d-47ae-8fad-a927fa4ccd58" class="plotly-graph-div" style="height:500px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("aa8f989d-1d5d-47ae-8fad-a927fa4ccd58")) {                    Plotly.newPlot(                        "aa8f989d-1d5d-47ae-8fad-a927fa4ccd58",                        [{"alignmentgroup": "True", "customdata": [[11729], [11729], [11729], [11729], [11729], [11729], [11729], [11729], [11729], [11729], [10311], [10311], [10311], [10311], [10311], [10311], [10311], [10311], [10311], [10311], [7988], [7988], [7988], [7988], [7988], [7988], [7988], [7988], [7988], [7988], [4736], [4736], [4736], [4736], [4736], [4736], [4736], [4736], [4736], [4736], [3481], [3481], [3481], [3481], [3481], [3481], [3481], [3481], [3481], [3481], [4838], [4838], [4838], [4838], [4838], [4838], [4838], [4838], [4838], [4838], [5848], [5848], [5848], [5848], [5848], [5848], [5848], [5848], [5848], [5848], [6106], [6106], [6106], [6106], [6106], [6106], [6106], [6106], [6106], [6106], [1559], [1559], [1559], [1559], [1559], [1559], [1559], [1559], [1559], [1559], [5342], [5342], [5342], [5342], [5342], [5342], [5342], [5342], [5342], [5342], [3781], [3781], [3781], [3781], [3781], [3781], [3781], [3781], [3781], [3781], [4408], [4408], [4408], [4408], [4408], [4408], [4408], [4408], [4408], [4408], [7292], [7292], [7292], [7292], [7292], [7292], [7292], [7292], [7292], [7292], [6044], [6044], [6044], [6044], [6044], [6044], [6044], [6044], [6044], [6044], [8181], [8181], [8181], [8181], [8181], [8181], [8181], [8181], [8181], [8181], [2352], [2352], [2352], [2352], [2352], [2352], [2352], [2352], [2352], [2352], [9491], [9491], [9491], [9491], [9491], [9491], [9491], [9491], [9491], [9491], [5847], [5847], [5847], [5847], [5847], [5847], [5847], [5847], [5847], [5847], [7939], [7939], [7939], [7939], [7939], [7939], [7939], [7939], [7939], [7939], [6078], [6078], [6078], [6078], [6078], [6078], [6078], [6078], [6078], [6078], [10268], [10268], [10268], [10268], [10268], [10268], [10268], [10268], [10268], [10268], [10060], [10060], [10060], [10060], [10060], [10060], [10060], [10060], [10060], [10060], [6256], [6256], [6256], [6256], [6256], [6256], [6256], [6256], [6256], [6256], [8436], [8436], [8436], [8436], [8436], [8436], [8436], [8436], [8436], [8436], [5085], [5085], [5085], [5085], [5085], [5085], [5085], [5085], [5085], [5085], [8783], [8783], [8783], [8783], [8783], [8783], [8783], [8783], [8783], [8783], [6047], [6047], [6047], [6047], [6047], [6047], [6047], [6047], [6047], [6047], [8832], [8832], [8832], [8832], [8832], [8832], [8832], [8832], [8832], [8832], [1961], [1961], [1961], [1961], [1961], [1961], [1961], [1961], [1961], [1961], [7845], [7845], [7845], [7845], [7845], [7845], [7845], [7845], [7845], [7845], [2699], [2699], [2699], [2699], [2699], [2699], [2699], [2699], [2699], [2699], [6004], [6004], [6004], [6004], [6004], [6004], [6004], [6004], [6004], [6004], [2868], [2868], [2868], [2868], [2868], [2868], [2868], [2868], [2868], [2868], [1875], [1875], [1875], [1875], [1875], [1875], [1875], [1875], [1875], [1875], [5593], [5593], [5593], [5593], [5593], [5593], [5593], [5593], [5593], [5593], [7849], [7849], [7849], [7849], [7849], [7849], [7849], [7849], [7849], [7849], [11223], [11223], [11223], [11223], [11223], [11223], [11223], [11223], [11223], [11223], [6170], [6170], [6170], [6170], [6170], [6170], [6170], [6170], [6170], [6170], [5168], [5168], [5168], [5168], [5168], [5168], [5168], [5168], [5168], [5168], [2866], [2866], [2866], [2866], [2866], [2866], [2866], [2866], [2866], [2866], [3437], [3437], [3437], [3437], [3437], [3437], [3437], [3437], [3437], [3437], [6929], [6929], [6929], [6929], [6929], [6929], [6929], [6929], [6929], [6929], [8894], [8894], [8894], [8894], [8894], [8894], [8894], [8894], [8894], [8894], [9177], [9177], [9177], [9177], [9177], [9177], [9177], [9177], [9177], [9177], [10349], [10349], [10349], [10349], [10349], [10349], [10349], [10349], [10349], [10349], [3988], [3988], [3988], [3988], [3988], [3988], [3988], [3988], [3988], [3988], [1549], [1549], [1549], [1549], [1549], [1549], [1549], [1549], [1549], [1549], [9541], [9541], [9541], [9541], [9541], [9541], [9541], [9541], [9541], [9541], [1181], [1181], [1181], [1181], [1181], [1181], [1181], [1181], [1181], [1181], [7790], [7790], [7790], [7790], [7790], [7790], [7790], [7790], [7790], [7790], [5663], [5663], [5663], [5663], [5663], [5663], [5663], [5663], [5663], [5663], [4601], [4601], [4601], [4601], [4601], [4601], [4601], [4601], [4601], [4601], [1603], [1603], [1603], [1603], [1603], [1603], [1603], [1603], [1603], [1603], [2212], [2212], [2212], [2212], [2212], [2212], [2212], [2212], [2212], [2212], [9761], [9761], [9761], [9761], [9761], [9761], [9761], [9761], [9761], [9761], [5428], [5428], [5428], [5428], [5428], [5428], [5428], [5428], [5428], [5428], [1098], [1098], [1098], [1098], [1098], [1098], [1098], [1098], [1098], [1098], [5939], [5939], [5939], [5939], [5939], [5939], [5939], [5939], [5939], [5939], [7333], [7333], [7333], [7333], [7333], [7333], [7333], [7333], [7333], [7333], [8358], [8358], [8358], [8358], [8358], [8358], [8358], [8358], [8358], [8358], [10650], [10650], [10650], [10650], [10650], [10650], [10650], [10650], [10650], [10650], [6083], [6083], [6083], [6083], [6083], [6083], [6083], [6083], [6083], [6083], [1424], [1424], [1424], [1424], [1424], [1424], [1424], [1424], [1424], [1424], [9281], [9281], [9281], [9281], [9281], [9281], [9281], [9281], [9281], [9281], [7161], [7161], [7161], [7161], [7161], [7161], [7161], [7161], [7161], [7161], [7185], [7185], [7185], [7185], [7185], [7185], [7185], [7185], [7185], [7185], [12980], [12980], [12980], [12980], [12980], [12980], [12980], [12980], [12980], [12980], [8282], [8282], [8282], [8282], [8282], [8282], [8282], [8282], [8282], [8282], [3962], [3962], [3962], [3962], [3962], [3962], [3962], [3962], [3962], [3962], [720], [720], [720], [720], [720], [720], [720], [720], [720], [720], [11723], [11723], [11723], [11723], [11723], [11723], [11723], [11723], [11723], [11723], [8446], [8446], [8446], [8446], [8446], [8446], [8446], [8446], [8446], [8446], [8790], [8790], [8790], [8790], [8790], [8790], [8790], [8790], [8790], [8790], [1241], [1241], [1241], [1241], [1241], [1241], [1241], [1241], [1241], [1241], [9098], [9098], [9098], [9098], [9098], [9098], [9098], [9098], [9098], [9098], [1739], [1739], [1739], [1739], [1739], [1739], [1739], [1739], [1739], [1739], [4636], [4636], [4636], [4636], [4636], [4636], [4636], [4636], [4636], [4636], [8421], [8421], [8421], [8421], [8421], [8421], [8421], [8421], [8421], [8421], [5215], [5215], [5215], [5215], [5215], [5215], [5215], [5215], [5215], [5215], [7062], [7062], [7062], [7062], [7062], [7062], [7062], [7062], [7062], [7062], [6714], [6714], [6714], [6714], [6714], [6714], [6714], [6714], [6714], [6714], [7630], [7630], [7630], [7630], [7630], [7630], [7630], [7630], [7630], [7630], [6970], [6970], [6970], [6970], [6970], [6970], [6970], [6970], [6970], [6970], [3146], [3146], [3146], [3146], [3146], [3146], [3146], [3146], [3146], [3146], [5860], [5860], [5860], [5860], [5860], [5860], [5860], [5860], [5860], [5860], [7755], [7755], [7755], [7755], [7755], [7755], [7755], [7755], [7755], [7755], [6105], [6105], [6105], [6105], [6105], [6105], [6105], [6105], [6105], [6105], [4030], [4030], [4030], [4030], [4030], [4030], [4030], [4030], [4030], [4030], [3859], [3859], [3859], [3859], [3859], [3859], [3859], [3859], [3859], [3859], [4268], [4268], [4268], [4268], [4268], [4268], [4268], [4268], [4268], [4268], [9409], [9409], [9409], [9409], [9409], [9409], [9409], [9409], [9409], [9409], [1979], [1979], [1979], [1979], [1979], [1979], [1979], [1979], [1979], [1979], [12658], [12658], [12658], [12658], [12658], [12658], [12658], [12658], [12658], [12658], [5394], [5394], [5394], [5394], [5394], [5394], [5394], [5394], [5394], [5394], [1210], [1210], [1210], [1210], [1210], [1210], [1210], [1210], [1210], [1210], [6971], [6971], [6971], [6971], [6971], [6971], [6971], [6971], [6971], [6971], [2056], [2056], [2056], [2056], [2056], [2056], [2056], [2056], [2056], [2056], [2782], [2782], [2782], [2782], [2782], [2782], [2782], [2782], [2782], [2782], [4974], [4974], [4974], [4974], [4974], [4974], [4974], [4974], [4974], [4974], [6966], [6966], [6966], [6966], [6966], [6966], [6966], [6966], [6966], [6966]], "hovertemplate": "max_deph=%{x}<br>test_roc_auc=%{y}<br>store_id=%{customdata[0]}<extra></extra>", "legendgroup": "", "marker": {"color": "#0b6fab"}, "name": "", "notched": false, "offsetgroup": "", "orientation": "v", "showlegend": false, "type": "box", "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], "x0": " ", "xaxis": "x", "y": [0.7998009415320237, 0.792223642915262, 0.7671306520981053, 0.7673994536789492, 0.7626917935603859, 0.7787181797047541, 0.7810211554109032, 0.7755652098105313, 0.7849369406021156, 0.7932915843310473, 0.8472782258064516, 0.8612021169354839, 0.8675655241935484, 0.887600806451613, 0.748991935483871, 0.7669480846774194, 0.7489289314516129, 0.7723034274193548, 0.7518271169354839, 0.8017893145161292, 0.5788797061524334, 0.5437557392102847, 0.5428833792470156, 0.5755280073461893, 0.5551652892561983, 0.5645546372819099, 0.5682277318640955, 0.5662993572084482, 0.5751147842056934, 0.5839990817263545, 0.881456953642384, 0.9000551876379691, 0.8919977924944812, 0.8893487858719646, 0.8963576158940396, 0.9080573951434878, 0.8453090507726269, 0.6830022075055189, 0.7480132450331127, 0.7239514348785873, 0.7698193356875992, 0.7819873074364092, 0.7568580787143661, 0.7672347612467374, 0.7590332156200419, 0.7209427299247656, 0.7125876452223756, 0.6955831925891807, 0.7377040790214442, 0.7167843799580327, 0.9725621118800871, 0.9418784695499278, 0.9401342375350084, 0.9677281533960878, 0.9754980592644282, 0.9716679445321742, 0.9673532806083847, 0.9661870983700143, 0.9694605618304354, 0.971218503395834, 0.9512195121951219, 0.9749128919860627, 0.9639953542392566, 0.9593495934959351, 0.9282229965156794, 0.9392566782810685, 0.9041811846689896, 0.9455284552845529, 0.819163763066202, 0.9420441347270616, 0.7122153209109732, 0.7142857142857143, 0.7142857142857143, 0.7122153209109732, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.9361882716049383, 0.9341820987654321, 0.9137345679012345, 0.9118055555555555, 0.8662037037037037, 0.8688271604938271, 0.8861882716049383, 0.8215277777777776, 0.8554012345679012, 0.9070216049382716, 0.9485429179954697, 0.9454214043430735, 0.93691612481547, 0.9335805001183246, 0.9208634309604571, 0.9159332424300477, 0.9006496579857786, 0.8910371989767746, 0.8862873144840486, 0.86008406675757, 0.9681682182034642, 0.9721449658006827, 0.9669798221891834, 0.9639906751143181, 0.9614800194244717, 0.958896023537219, 0.9648088099378103, 0.9580501191244176, 0.9587771127317158, 0.9573110208243438, 0.7622395833333333, 0.7755208333333334, 0.7807291666666668, 0.7553385416666666, 0.7076822916666666, 0.6484375, 0.6473958333333334, 0.6497395833333333, 0.6497395833333334, 0.65, 0.4882075471698113, 0.49410377358490565, 0.49410377358490565, 0.49410377358490565, 0.49410377358490565, 0.49410377358490565, 0.49410377358490565, 0.49410377358490565, 0.49410377358490565, 0.49410377358490565, 0.9391182786744848, 0.9609643849212395, 0.9621332522268733, 0.9664695366708844, 0.9601046598301639, 0.9516141234916939, 0.9514175351083326, 0.9571796988770342, 0.9605555521704779, 0.9605884720504296, 0.4708935018050542, 0.5047382671480145, 0.4984205776173285, 0.5045126353790614, 0.4984205776173285, 0.4979693140794224, 0.4984205776173285, 0.4984205776173285, 0.4981949458483754, 0.4984205776173285, 0.925958025547348, 0.9238479692791807, 0.9343749166644444, 0.8902687404997467, 0.9012740339742393, 0.8851402704072108, 0.8969805861489639, 0.8836252300061335, 0.8583745566548442, 0.9010290274407318, 0.9478134396763709, 0.9447192090567468, 0.9456825200323359, 0.9497752387137363, 0.9451541872571005, 0.9443180549777267, 0.9468812468924089, 0.9443183932189401, 0.9425666419750582, 0.9451802318305276, 0.9746899634790255, 0.9739173206924621, 0.9706265153592679, 0.9686959966221362, 0.9672247106398352, 0.9633941435853237, 0.9593589894267643, 0.9594438713103586, 0.9600706913738242, 0.9556764215539043, 0.9141670150501672, 0.9190618032329989, 0.9070948299888517, 0.9151076505016723, 0.9209866220735785, 0.9164663461538461, 0.9256288322185061, 0.9302536231884058, 0.9068030588071347, 0.917441819955407, 0.4688888888888889, 0.4688888888888889, 0.4822222222222222, 0.47333333333333333, 0.47333333333333333, 0.47333333333333333, 0.47333333333333333, 0.47333333333333333, 0.47333333333333333, 0.47333333333333333, 0.5678763440860215, 0.529233870967742, 0.5366263440860215, 0.5406586021505376, 0.5386424731182796, 0.5379704301075269, 0.5379704301075269, 0.5359543010752688, 0.5372983870967742, 0.5379704301075269, 0.8800815150556903, 0.8782544534626332, 0.8684867010997506, 0.8613892695267208, 0.8644636520150379, 0.8680299357014862, 0.8760233301711113, 0.866993429605425, 0.8589473314360001, 0.8513755665647729, 0.9011628810845265, 0.9051361321544615, 0.8655157786261745, 0.8750435858313288, 0.89367167043115, 0.9008219677031766, 0.8965255707245354, 0.8033784849231888, 0.8944767590353148, 0.8936672285629892, 0.6931181648439585, 0.7264603894371833, 0.6925846892504667, 0.6715124033075487, 0.6613763670312083, 0.6728460922912777, 0.6735129367831422, 0.6129634569218458, 0.6360362763403574, 0.6355028007468658, 0.9007142857142857, 0.6773809523809523, 0.6792857142857142, 0.6314285714285713, 0.6323809523809523, 0.6319047619047617, 0.6314285714285715, 0.6333333333333333, 0.6323809523809524, 0.6309523809523809, 0.8942032500606354, 0.8609750181906378, 0.8490096208262592, 0.8393564556552671, 0.8503355162098796, 0.8574581615328644, 0.8588163958282804, 0.8449753415797558, 0.8440617673215296, 0.824626081332363, 0.9217818946989098, 0.9188733092238196, 0.9100027670171555, 0.9153976461049407, 0.9155923025443687, 0.9125622578859989, 0.9015963436764005, 0.9041212468308, 0.9071207256019871, 0.9122606208414306, 0.9278867754979782, 0.9226074584394188, 0.9047850831211622, 0.9228133892466677, 0.9259585143028306, 0.9184326793470121, 0.918301632469672, 0.8938145873895463, 0.9109255653736709, 0.8899767859817284, 0.9182456636500754, 0.9161246229260936, 0.9113876319758673, 0.9093372926093515, 0.9107513197586727, 0.8977187028657617, 0.8740101809954752, 0.870439762443439, 0.8727493401206636, 0.8825179110105581, 0.8489583333333334, 0.7291666666666667, 0.7296772875816995, 0.7291666666666667, 0.7100694444444444, 0.7365196078431372, 0.6496119281045751, 0.652062908496732, 0.6894403594771242, 0.688827614379085, 0.9727720594703968, 0.9749186240872701, 0.966992170317586, 0.9687120612298759, 0.9631433095803641, 0.9572622503738895, 0.9601258027623822, 0.9688572182633941, 0.9482581155977831, 0.94828010908771, 0.8930093094462344, 0.8826928245095405, 0.8702777131917541, 0.8807557470656976, 0.8527926828900048, 0.8840697649840414, 0.9007047761112905, 0.8853681956946453, 0.8971612681138633, 0.8843007648662767, 0.9719036697247706, 0.9476529051987768, 0.9642660550458716, 0.961743119266055, 0.9331192660550459, 0.9336085626911315, 0.9348012232415901, 0.934006116207951, 0.9337538226299694, 0.9349082568807339, 0.9532228090766822, 0.9522447183098592, 0.9350058685446009, 0.9214593114241002, 0.9301643192488264, 0.9470852895148669, 0.9426838810641627, 0.918671752738654, 0.8153853677621283, 0.9316803599374022, 0.9397005962671559, 0.9337822390008116, 0.930616942454927, 0.9290034223617825, 0.929771019299298, 0.9290584624069436, 0.9354292770701761, 0.92449239671171, 0.9323051194298416, 0.9333106587164379, 0.8871021275183378, 0.8704404394437253, 0.8727803777091838, 0.8833432905174415, 0.8641508181486276, 0.8493643997477512, 0.8765143217498091, 0.87624049918683, 0.8938398220983106, 0.8822480002655249, 0.7540815579471412, 0.7064206750128121, 0.704077897357054, 0.6836518046709129, 0.6837250164726554, 0.7070063694267514, 0.730800204993045, 0.7173658393733069, 0.687641847865876, 0.7165605095541401, 0.9756644435240857, 0.9767149220313271, 0.9762198324629596, 0.9763261497741154, 0.9756977668604181, 0.9758262997291289, 0.9750487552147055, 0.971589475538291, 0.9680960791127742, 0.9677152409832606, 0.6966783216783217, 0.541083916083916, 0.5585664335664335, 0.5734265734265734, 0.5856643356643356, 0.5856643356643356, 0.5874125874125874, 0.5891608391608392, 0.5856643356643356, 0.5856643356643356, 0.9452331786498388, 0.9420277159970559, 0.9433950543393099, 0.9388030589571306, 0.9382146637440013, 0.9394842852062576, 0.9416173051010654, 0.9358896999777694, 0.9445142616041412, 0.9412721551213591, 0.9626409017713365, 0.9487922705314009, 0.9544283413848631, 0.9542673107890499, 0.9185185185185185, 0.9054750402576489, 0.8735909822866345, 0.9156199677938809, 0.9231884057971013, 0.9225442834138486, 0.9356613756613756, 0.9223280423280423, 0.8753439153439153, 0.8962811791383221, 0.8106122448979594, 0.8105215419501135, 0.8045351473922904, 0.796931216931217, 0.8375812547241118, 0.8072260015117159, 0.8900672330424397, 0.884586487892273, 0.8953649656128995, 0.8850295709799842, 0.8878229208807722, 0.8792598585987016, 0.844159972259146, 0.8609586006280221, 0.8392186326070623, 0.8341520738214954, 0.9387868549481648, 0.9368087441162031, 0.9332423691125935, 0.931376158710907, 0.9239004880302637, 0.9194352998931532, 0.92035216148315, 0.9119740679777065, 0.9076027318143753, 0.9133349215975051, 0.8273926917835468, 0.795335658238884, 0.8044286064956281, 0.8092596981572866, 0.7984526963543562, 0.7396902853370126, 0.8019929575676522, 0.7753828052919817, 0.7990981115785375, 0.7753616441370904, 0.9831115779645191, 0.9875583566760038, 0.9821778711484594, 0.9745098039215686, 0.9758403361344539, 0.9736928104575163, 0.9293067226890755, 0.9259103641456583, 0.9287931839402428, 0.9197478991596639, null, null, null, null, null, null, null, null, null, null, 0.9390042291426375, 0.9543204536716647, 0.9411524413687042, 0.9390234525182621, 0.9549980776624375, 0.9295174932718185, 0.9515618992695117, 0.9251922337562476, 0.9357026143790849, 0.9360438292964244, 0.983908269190029, 0.9628733846479769, 0.9787841607231128, 0.9762949297366006, 0.9762861026763647, 0.9772659063625451, 0.9791902054939623, 0.9614434008897677, 0.9681475531389026, 0.9636104441776712, 0.7021129541864138, 0.7021129541864138, 0.7005331753554503, 0.6992496050552923, 0.7017180094786729, 0.7021129541864138, 0.7007306477093208, 0.699545813586098, 0.7019154818325434, 0.7013230647709321, 0.953384321122168, 0.9514109857117948, 0.943820201514174, 0.9438622438698557, 0.9459733601932355, 0.9327940405477131, 0.9427627993895155, 0.9401738753817371, 0.9316057908108096, 0.9439309868093216, 0.9750627297144473, 0.9726772335312412, 0.9702608142493638, 0.9659271628498729, 0.9698190556969184, 0.9611517528979361, 0.9554619027424371, 0.934222151540854, 0.938295165394402, 0.9396601993214589, 0.970880304773564, 0.9707410907765, 0.9707082650842322, 0.9711573158313356, 0.9688150019553723, 0.9661947323973997, 0.9645592336074633, 0.9658083810840887, 0.9666142636370678, 0.9659879777673239, 0.8591588616714698, 0.86569704610951, 0.895560158501441, 0.9041066282420749, 0.9154628962536022, 0.917246037463977, 0.8202629682997119, 0.8211545389048991, 0.892435158501441, 0.8855817723342939, 0.6725605844618674, 0.5492106200997862, 0.6164700641482538, 0.7388720598717035, 0.5927387740555952, 0.46732715609408415, 0.6566411261582323, 0.6100053456878118, 0.4339272986457591, 0.5264754098360656, 0.9352708342296605, 0.913264208578927, 0.8941186593813192, 0.8941724390138966, 0.9097577765348708, 0.8837284343673365, 0.9152432990577809, 0.9182442025556083, 0.9027986920793356, 0.9007012864088113, 0.9681116296724166, 0.9609669416183444, 0.9460296591867053, 0.9592216697989029, 0.9616150448140128, 0.9540413722935491, 0.9636526718113303, 0.9588222161585593, 0.9637255145155821, 0.9607728256551218, 0.9713447653429602, 0.7348826714801444, 0.9896209386281589, 0.5038357400722021, 0.5015794223826715, 0.5027075812274368, 0.5022563176895307, 0.5051895306859205, 0.5022563176895307, 0.5047382671480144, 0.956977483990911, 0.9597868209047717, 0.9542309440198307, 0.9471695930592853, 0.9495740549473249, 0.9518347448874199, 0.9457203057219582, 0.9473505474075604, 0.9499516628795703, 0.950965502995249, 0.8264941857713122, 0.90841800911974, 0.8769919847268022, 0.876917038766784, 0.75987708862557, 0.7267884472774895, 0.7271730383881096, 0.8091737799586612, 0.7922241594218905, 0.8249439877561969, 0.8954954231572185, 0.8599780525667791, 0.8716610459825492, 0.866763021251539, 0.849686847599165, 0.8747658048284354, 0.8687436432739147, 0.8322894919972165, 0.8274048498474387, 0.8313527113109577, 0.8792508233800246, 0.7033351851117019, 0.6719475417642157, 0.6285663267330662, 0.6351533669298837, 0.6308083012578867, 0.6376235070036905, 0.6347664775207333, 0.4357664378397684, 0.6441014245466451, 0.9582307925255051, 0.957593300003767, 0.9570024412402309, 0.9565617645621928, 0.9548651017792052, 0.9554821971721356, 0.9531191733504274, 0.9521964500113254, 0.9514573830779826, 0.9501830560064035, 0.8780815007995536, 0.8573253056011676, 0.8078012814321132, 0.8204331541045536, 0.8322547409768506, 0.8386619015422262, 0.8491043927149404, 0.8594395612651191, 0.8759618790044754, 0.8623909333848482, 0.8373688458434222, 0.8307102502017757, 0.7974172719935432, 0.6244955609362389, 0.5996771589991929, 0.7203389830508475, 0.6620258272800646, 0.6049233252623083, 0.609362389023406, 0.6065375302663438, 0.542584058713091, 0.7201236636720508, 0.6774775403807661, 0.6819966174804885, 0.7125246318794707, 0.6591025461993204, 0.664990923055439, 0.6667015779919006, 0.6755574174929012, 0.6769112011047494, 0.9166590389016018, 0.9104805491990846, 0.9085354691075515, 0.9052402745995424, 0.9084668192219678, 0.8996796338672769, 0.8642562929061786, 0.8719221967963386, 0.8642334096109839, 0.8653318077803205, 0.9084970596097302, 0.9271251002405774, 0.9167502004811547, 0.9140102913659449, 0.8967689120556, 0.8772052927024859, 0.8938285217856189, 0.8523957497995189, 0.8844894413258488, 0.8800955626837743, 0.9338517658300869, 0.9454551948045349, 0.9609740524363579, 0.9612289157081287, 0.9670878673613428, 0.9661913090390298, 0.9577456875406776, 0.9655573811118369, 0.9623671985234628, 0.9613025219051026, 0.8236751280338455, 0.8354904252950346, 0.7940603429080383, 0.8244962146515252, 0.8289217323535961, 0.8143787575150301, 0.8160487641950568, 0.8101341572032955, 0.8117345802716545, 0.7774716098864396, 0.9097447795823665, 0.8560614849187935, 0.8990477571539055, 0.9032095901005414, 0.9066028615622584, 0.9098269528228925, 0.912195475638051, 0.9238930781129157, 0.8985837200309359, 0.9072409126063419, 0.49235181644359466, 0.494263862332696, 0.4933078393881453, 0.497131931166348, 0.498565965583174, 0.497131931166348, 0.49569789674952197, 0.497131931166348, 0.497131931166348, 0.49665391969407263, 0.812192118226601, 0.8172208538587848, 0.8118329228243021, 0.8094724958949097, 0.7797105911330049, 0.7216235632183908, 0.7934113300492611, 0.7718596059113301, 0.7907430213464696, 0.8003899835796388, 0.8878205128205128, 0.8790064102564104, 0.8657852564102564, 0.7884615384615384, 0.742588141025641, 0.7377804487179487, 0.7405849358974358, 0.7393830128205128, 0.7393830128205128, 0.7369791666666666, 0.90170054381742, 0.9015110992243915, 0.9029375055718998, 0.8959169118302576, 0.8775519301060889, 0.8454020682892038, 0.7927810466256575, 0.8024092894713382, 0.7432245698493358, 0.7854595702950877, 0.8907914764079148, 0.8192541856925417, 0.8116438356164384, 0.8116438356164384, 0.8105022831050228, 0.8116438356164384, 0.8116438356164384, 0.8116438356164384, 0.6529680365296804, 0.8120243531202436, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.625, 0.5, 0.625, 0.5, 0.4256342957130359, 0.4658792650918635, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8820181068846698, 0.9006654955746233, 0.8865166768011171, 0.8951703713713036, 0.8901369276850662, 0.889055919645069, 0.879146679278427, 0.8154122469202533, 0.8627513906718015, 0.8680888678692881, null, null, null, null, null, null, null, null, null, null, 0.6786191239316239, 0.61502849002849, 0.6477029914529915, 0.4994212962962963, 0.6150952635327636, 0.6153846153846154, 0.6145165598290598, 0.6145165598290598, 0.6532674501424501, 0.5, 0.7404761904761904, 0.6158730158730159, 0.6142857142857142, 0.6142857142857142, 0.6142857142857142, 0.6158730158730159, 0.6158730158730159, 0.6158730158730159, 0.6142857142857142, 0.6142857142857142, 0.8997935898556474, 0.9036528544438692, 0.9034175334323922, 0.9027989753450815, 0.8991683083108658, 0.8976824242098257, 0.8929491101504037, 0.8914968433367175, 0.8841279339487538, 0.8626632959733214, 0.6939873417721518, 0.7109177215189874, 0.761867088607595, 0.7639240506329114, 0.7678797468354431, 0.7837025316455696, 0.7832278481012658, 0.6996835443037975, 0.6575949367088608, 0.8808544303797469, null, null, null, null, null, null, null, null, null, null, 0.8633267195767196, 0.8540013227513228, 0.853207671957672, 0.853042328042328, 0.8377976190476191, 0.848015873015873, 0.8377645502645503, 0.8362103174603175, 0.8434854497354498, 0.8417328042328043, 0.9737184037038038, 0.9738107738989138, 0.9706744711003111, 0.9709695202452708, 0.9714913449127183, 0.9702377876561825, 0.9687788740528375, 0.9644918260409518, 0.9617785519619831, 0.9541953605530002, 0.9042661528430029, 0.8627841720820848, 0.8864117573794993, 0.8966613672496025, 0.9096034228861554, 0.9228275442696695, 0.9095887701200794, 0.8745503432410454, 0.8422739627673214, 0.8704182632076369, 0.9677238534064282, 0.9668115673942532, 0.9660254038722407, 0.9671044518436305, 0.9664780434758237, 0.9654592540275116, 0.9631820423995784, 0.9610491709548314, 0.9638673079294611, 0.9648006143566631, 0.8855541524088859, 0.8655457335288687, 0.8658784150779426, 0.8721925750909781, 0.8744670305795449, 0.88588683395796, 0.8744195046439629, 0.8785203139427517, 0.8775596111020586, 0.8769926945847591, 0.9690199607108791, 0.9587599337435182, 0.9700886572253882, 0.9540784755604914, 0.9485580027940131, 0.9565766043282885, 0.9592125422547199, 0.948233745950167, 0.9469461760660618, 0.9393437041480556, 0.4996967491508976, 0.48891112728448977, 0.7951085638039788, 0.7106026605207828, 0.34546336729742844, 0.4262493934983018, 0.3402120734271389, 0.34209728287239205, 0.5394327187449457, 0.4242277211709526, 0.7802083333333333, 0.7374999999999999, 0.73671875, 0.7419270833333333, 0.8091145833333333, 0.8244791666666667, 0.8270833333333334, 0.8278645833333333, 0.82578125, 0.8197916666666667, 0.8688614842545695, 0.8665492182338692, 0.8658885707993834, 0.8255890772957497, 0.8676502972913456, 0.8696322395948028, 0.8192028187623871, 0.8707333186522792, 0.8720546135212508, 0.8210746531600969, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9178885630498534, 0.6546920821114369, 0.3425219941348973, 0.3425219941348973, 0.3425219941348973, 0.3425219941348973, 0.3425219941348973, 0.3425219941348973, 0.3450146627565982, 0.3425219941348973, 0.9634246963625632, 0.9673337371478272, 0.9599136458236263, 0.9649671512440592, 0.9589846783462244, 0.9594040241667443, 0.95385885518591, 0.9504584653496102, 0.9485811162364488, 0.9457262642499923, 0.5665928840793573, 0.6916133792424989, 0.6942039678635842, 0.7004508935891129, 0.7008444007214297, 0.6950319724545007, 0.698893261190359, 0.6980324643384161, 0.699139203148057, 0.698893261190359, 0.8683673469387754, 0.8678571428571429, 0.8821428571428571, 0.8056122448979591, 0.6535714285714285, 0.5903061224489796, 0.5872448979591837, 0.5872448979591837, 0.5872448979591837, 0.6698979591836735, 0.8241905145432117, 0.8186278507855271, 0.8118596246189608, 0.8050608133595686, 0.8049652349444881, 0.8027554619878271, 0.7967123573970046, 0.7922354644346346, 0.7981243691824605, 0.8070450212566396], "y0": " ", "yaxis": "y"}],                        {"boxmode": "group", "height": 500, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Distribution of test_roc_auc by max depth value"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "max_deph"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "test_roc_auc"}}},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('aa8f989d-1d5d-47ae-8fad-a927fa4ccd58');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


<a id='reference12'></a>
#### Frequency of best hyper-parameter values


```python
plt.figure(figsize=(8,5))

best_param_freq = sns.countplot(best_max_dep_values['param_value'], palette='Greens_r')

for p in best_param_freq.patches:
    best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

plt.ylabel('frequency')
plt.xlabel('max_depth')
plt.title('Count of best max depth value', loc='left')
```




    Text(0.0, 1.0, 'Count of best max depth value')




![png](output_150_1.png)



```python
plt.figure(figsize=(12,5))

best_param_freq = sns.countplot(best_max_dep_values[best_max_dep_values.param_value.isin([1,3,5,7])]['param_value'], palette='Greens',
                                hue=best_max_dep_values[best_max_dep_values.param_value.isin([1,3,5,7])]['quartile_n_orders'])

for p in best_param_freq.patches:
    best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

plt.ylabel('frequency')
plt.xlabel('max_depth')
plt.legend(loc='upper left', title='quartile_n_orders', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
plt.title('Count of best max depth value', loc='left')
```




    Text(0.0, 1.0, 'Count of best max depth value')




![png](output_151_1.png)



```python
# plt.figure(figsize=(12,5))

# best_param_freq = sns.countplot(best_max_dep_values[best_max_dep_values.param_value.isin([1,3,5,7])]['quartile_n_orders'],
#                                 palette='Greens_r',
#                                 hue=best_max_dep_values[best_max_dep_values.param_value.isin([1,3,5,7])]['param_value'])

# for p in best_param_freq.patches:
#     best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                              ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

# plt.ylabel('frequency')
# plt.xlabel('quartile_n_orders')
# plt.legend(loc='upper left', title='max_depth', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
# plt.title('Count of best max depth value', loc='left')
```


```python
plt.figure(figsize=(12,5))

best_param_freq = sns.countplot(best_max_dep_values[best_max_dep_values.param_value.isin([1,3,5,7])]['param_value'],
                                palette='Greens',
                                hue=best_max_dep_values[best_max_dep_values.param_value.isin([1,3,5,7])]['quartile_n_vars'])

for p in best_param_freq.patches:
    best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

plt.ylabel('frequency')
plt.xlabel('max_depth')
plt.legend(loc='upper left', title='quartile_n_vars', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
plt.title('Count of best max depth value', loc='left')
```




    Text(0.0, 1.0, 'Count of best max depth value')




![png](output_153_1.png)



```python
# plt.figure(figsize=(12,5))

# best_param_freq = sns.countplot(best_max_dep_values[best_max_dep_values.param_value.isin([1,3,5,7])]['quartile_n_vars'],
#                                 palette='Greens_r',
#                                 hue=best_max_dep_values[best_max_dep_values.param_value.isin([1,3,5,7])]['param_value'])

# for p in best_param_freq.patches:
#     best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                              ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

# plt.ylabel('frequency')
# plt.xlabel('quartile_n_vars')
# plt.legend(loc='upper left', title='max_depth', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
# plt.title('Count of best max depth value', loc='left')
```

[(Main conclusion)](#main_conclusions)<a href='#main_conclusions'></a>

#### Performance metric against dataset information

When exploring max depth, the plot of performance metrics against numbers of observations and features produce the same patters as those obtained when exploring [subsample](#performance_data_info)<a href='#performance_data_info'></a>.

<a id='reference13'></a>
#### Distribution of performance metric by best hyper-parameter value


```python
px.strip(data_frame=best_max_dep_values.sort_values('param_value'),
         x=best_max_dep_values.sort_values('param_value')['param_value'],
         y=best_max_dep_values.sort_values('param_value')['test_roc_auc'],
         hover_data=['store_id', 'param_value'],
         color_discrete_sequence=['#0b6fab'],
         width=900, height=500, title='Distribution of test_roc_auc by best max depth value',
         labels={'y': 'test_roc_auc', 'x': 'max_depth'})
```


<div>                            <div id="f41d38fb-beaf-4ecf-b2bd-f16a3b911f02" class="plotly-graph-div" style="height:500px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("f41d38fb-beaf-4ecf-b2bd-f16a3b911f02")) {                    Plotly.newPlot(                        "f41d38fb-beaf-4ecf-b2bd-f16a3b911f02",                        [{"alignmentgroup": "True", "boxpoints": "all", "customdata": [[12980.0, 1.0], [3859.0, 1.0], [8783.0, 1.0], [6083.0, 1.0], [6714.0, 1.0], [6966.0, 1.0], [6047.0, 1.0], [4601.0, 1.0], [7845.0, 1.0], [6929.0, 1.0], [7790.0, 1.0], [7630.0, 1.0], [5085.0, 1.0], [5168.0, 1.0], [5342.0, 1.0], [6971.0, 1.0], [5428.0, 1.0], [5593.0, 1.0], [5663.0, 1.0], [7755.0, 1.0], [5847.0, 1.0], [8832.0, 1.0], [9177.0, 1.0], [1098.0, 1.0], [1181.0, 1.0], [1210.0, 1.0], [1241.0, 1.0], [1424.0, 1.0], [1559.0, 1.0], [11729.0, 1.0], [1739.0, 1.0], [3437.0, 1.0], [1875.0, 1.0], [11223.0, 1.0], [10650.0, 1.0], [10349.0, 1.0], [10268.0, 1.0], [10060.0, 1.0], [9281.0, 1.0], [2866.0, 1.0], [2868.0, 1.0], [1961.0, 1.0], [7161.0, 1.0], [6970.0, 2.0], [8358.0, 2.0], [7292.0, 2.0], [7333.0, 2.0], [8181.0, 2.0], [8282.0, 2.0], [8436.0, 2.0], [8790.0, 2.0], [7185.0, 2.0], [6256.0, 2.0], [720.0, 2.0], [6106.0, 2.0], [6105.0, 2.0], [2056.0, 2.0], [2699.0, 2.0], [5848.0, 2.0], [5215.0, 2.0], [3481.0, 2.0], [3781.0, 2.0], [6170.0, 2.0], [3988.0, 2.0], [1979.0, 3.0], [2352.0, 3.0], [9409.0, 3.0], [9098.0, 3.0], [8894.0, 3.0], [8421.0, 3.0], [6078.0, 3.0], [5939.0, 3.0], [4974.0, 3.0], [4408.0, 3.0], [1603.0, 4.0], [6044.0, 4.0], [10311.0, 4.0], [9761.0, 4.0], [9491.0, 4.0], [8446.0, 5.0], [3962.0, 5.0], [4838.0, 5.0], [9541.0, 5.0], [2782.0, 5.0], [2212.0, 6.0], [4268.0, 6.0], [4736.0, 6.0], [4030.0, 6.0], [4636.0, 7.0], [6004.0, 7.0], [7939.0, 8.0], [12658.0, 8.0], [11723.0, 8.0], [7849.0, 9.0], [5394.0, 9.0], [7988.0, 10.0], [3146.0, 10.0]], "fillcolor": "rgba(255,255,255,0)", "hoveron": "points", "hovertemplate": "max_depth=%{x}<br>test_roc_auc=%{y}<br>store_id=%{customdata[0]}<br>param_value=%{customdata[1]}<extra></extra>", "legendgroup": "", "line": {"color": "rgba(255,255,255,0)"}, "marker": {"color": "#0b6fab"}, "name": "", "offsetgroup": "", "orientation": "v", "pointpos": 0, "showlegend": false, "type": "box", "x": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0], "x0": " ", "xaxis": "x", "y": [0.9166590389016018, 0.9677238534064282, 0.8942032500606354, 0.8792508233800246, 0.6786191239316239, 0.8241905145432117, 0.9217818946989098, 0.9750627297144473, 0.8489583333333334, 0.9356613756613756, 0.7021129541864138, 0.7404761904761904, 0.9007142857142857, 0.6966783216783217, 0.9485429179954697, 0.9178885630498534, 0.9352708342296605, 0.9397005962671559, 0.953384321122168, 0.8633267195767196, 0.9746899634790255, 0.9278867754979782, 0.9387868549481648, 0.9681116296724166, 0.983908269190029, 0.5, 0.8878205128205128, 0.9582307925255051, 0.9361882716049383, 0.7998009415320237, 0.8907914764079148, 0.9626409017713365, 0.9532228090766822, 0.7540815579471412, 0.8954954231572185, 0.8273926917835468, 0.5678763440860215, 0.8800815150556903, 0.8780815007995536, 0.9452331786498388, 0.9719036697247706, 0.9182456636500754, 0.8373688458434222, 0.9036528544438692, 0.90841800911974, 0.49410377358490565, 0.9597868209047717, 0.5047382671480145, 0.9271251002405774, 0.7264603894371833, 0.8172208538587848, 0.7201236636720508, 0.9051361321544615, 0.8354904252950346, 0.7142857142857143, 0.9738107738989138, 0.9673337371478272, 0.9749186240872701, 0.9749128919860627, 0.9006654955746233, 0.7819873074364092, 0.9721449658006827, 0.9767149220313271, 0.9875583566760038, 0.7951085638039788, 0.9343749166644444, 0.9700886572253882, 0.9029375055718998, 0.8953649656128995, 0.5, 0.4822222222222222, 0.9896209386281589, 0.8821428571428571, 0.7807291666666668, 0.9711573158313356, 0.9664695366708844, 0.887600806451613, 0.7388720598717035, 0.9497752387137363, 0.498565965583174, 0.9670878673613428, 0.9754980592644282, 0.9549980776624375, 0.7008444007214297, 0.917246037463977, 0.88588683395796, 0.9080573951434878, 0.9228275442696695, 0.625, 0.9007047761112905, 0.9302536231884058, 0.8278645833333333, 0.9238930781129157, 0.8938398220983106, 0.8720546135212508, 0.5839990817263545, 0.8808544303797469], "y0": " ", "yaxis": "y"}],                        {"boxmode": "group", "height": 500, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Distribution of test_roc_auc by best max depth value"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "max_depth"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "test_roc_auc"}}},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('f41d38fb-beaf-4ecf-b2bd-f16a3b911f02');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

<a id='reference15'></a>
#### Correlation between performance metric of best hyper-parameter value and dataset information


```python
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(best_max_dep_values[['test_roc_auc', 'n_orders', 'n_vars', 'avg_y']].corr(),
                            dtype=np.bool))

sns.heatmap(best_max_dep_values[['test_roc_auc', 'n_orders', 'n_vars', 'avg_y']].corr(),
            mask = mask, annot = True, cmap = 'viridis')
plt.title('Correlation between performance metric and dataset information')
plt.tight_layout()
```


![png](output_162_0.png)



```python
# Generate masks for the upper triangle:
mask1 = np.triu(np.ones_like(metrics_max_depth[metrics_max_depth.param_value==1][['test_roc_auc', 
                                                                                 'n_orders', 'n_vars',
                                                                                 'avg_y']].corr(), dtype=np.bool))
mask4 = np.triu(np.ones_like(metrics_max_depth[metrics_max_depth.param_value==4][['test_roc_auc', 
                                                                                 'n_orders', 'n_vars',
                                                                                 'avg_y']].corr(), dtype=np.bool))
mask10 = np.triu(np.ones_like(metrics_max_depth[metrics_max_depth.param_value==10][['test_roc_auc', 
                                                                                 'n_orders', 'n_vars',
                                                                                 'avg_y']].corr(), dtype=np.bool))
```


```python
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))

corr_matrices = sns.heatmap(metrics_max_depth[metrics_max_depth.param_value==1][['test_roc_auc', 'n_orders',
                                                                 'n_vars', 'avg_y']].corr(),
                            mask = mask1, annot = True, cmap = 'viridis', ax=axs[0])
sns.heatmap(metrics_max_depth[metrics_max_depth.param_value==4][['test_roc_auc', 'n_orders',
                                                                 'n_vars', 'avg_y']].corr(),
            mask = mask4, annot = True, cmap = 'viridis', ax=axs[1])
sns.heatmap(metrics_max_depth[metrics_max_depth.param_value==10][['test_roc_auc', 'n_orders',
                                                                 'n_vars', 'avg_y']].corr(),
            mask = mask10, annot = True, cmap = 'viridis', ax=axs[2])


axs[0].set_title('max_depth = 1', loc='left')
axs[1].set_title('max_depth = 4', loc='left')
axs[2].set_title('max_depth = 10', loc='left')

plt.tight_layout()
```


![png](output_164_0.png)


[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

<a id='learning_rate'></a>

## Learning rate

Since learning rates $v \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$ are high and since high learning rates demand less estimators to learn from the data, this high-level set of learning rate values was tested using different values of $n\_estimators$. For all values tested $\{100, 250, 500\}$, performance metrics behaved very similarly. Therefore, outcomes presented and discussed next are based on $n\_estimators = 100$.

Main findings are listed below:
* Small values of learning rate ($v \leq 0.1$) perform notably better than large values ($v > 0.1$).
* The smallest value testes ($v = 0.01$) presented the best results.

<a id='proc_data_learning_rate'></a>

### Processing data

#### Performance metrics


```python
# Assessing missing hyper-parameter values:
for s in tun_learning_rate['II100'].keys():
    for m in ['test_roc_auc', 'test_prec_avg', 'test_pr_auc', 'test_deviance', 'test_brier_score']:
        if len(tun_learning_rate['II100'][s][m].keys()) != 9:
            print('Missing hyper-parameter value for store ' + str(s) + ' and metric ' + m + '!')
```


```python
# Assessing missing hyper-parameter values:
for s in tun_learning_rate['II250'].keys():
    for m in ['test_roc_auc', 'test_prec_avg', 'test_pr_auc', 'test_deviance', 'test_brier_score']:
        if len(tun_learning_rate['II250'][s][m].keys()) != 9:
            print('Missing hyper-parameter value for store ' + str(s) + ' and metric ' + m + '!')
```


```python
# Assessing missing hyper-parameter values:
for s in tun_learning_rate['I500'].keys():
    for m in ['test_roc_auc', 'test_prec_avg', 'test_pr_auc', 'test_deviance', 'test_brier_score']:
        if len(tun_learning_rate['I500'][s][m].keys()) != 10:
            print('Missing hyper-parameter value for store ' + str(s) + ' and metric ' + m + '!')
```


```python
# Assessing missing hyper-parameter values:
for s in tun_learning_rate['II500'].keys():
    for m in ['test_roc_auc', 'test_prec_avg', 'test_pr_auc', 'test_deviance', 'test_brier_score']:
        if len(tun_learning_rate['II500'][s][m].keys()) != 9:
            print('Missing hyper-parameter value for store ' + str(s) + ' and metric ' + m + '!')
```


```python
param_value = []
n_estimators = []
stores = []
test_roc_auc = []
test_prec_avg = []
test_pr_auc = []
test_deviance = []
test_brier_score = []

for k in ['I500', 'II100']:
    # Loop over datasets:
    for s in tun_learning_rate[k].keys():
        # Loop over hyper-parameter values:
        for v in tun_learning_rate[k][s]['test_roc_auc'].keys():
            stores.append(int(s))
            param_value.append(float(v))
            n_estimators.append(int(''.join([l for l in k if l not in ['I','II']])))
            test_roc_auc.append(tun_learning_rate[k][s]['test_roc_auc'][v])
            test_prec_avg.append(tun_learning_rate[k][s]['test_prec_avg'][v])
            test_pr_auc.append(tun_learning_rate[k][s]['test_pr_auc'][v])
            test_deviance.append(tun_learning_rate[k][s]['test_deviance'][v])
            test_brier_score.append(tun_learning_rate[k][s]['test_brier_score'][v])

metrics_learning = pd.DataFrame(data={'store_id': stores, 'param_value': param_value,
                                      'n_estimators': n_estimators, 'test_roc_auc': test_roc_auc,
                                      'test_prec_avg': test_prec_avg, 'test_pr_auc': test_pr_auc,
                                      'test_deviance': test_deviance, 'test_brier_score': test_brier_score})

print('\033[1mShape of metrics_learning:\033[0m ' + str(metrics_learning.shape) + '.')
metrics_learning.head()
```

    [1mShape of metrics_learning:[0m (1900, 8).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>param_value</th>
      <th>n_estimators</th>
      <th>test_roc_auc</th>
      <th>test_prec_avg</th>
      <th>test_pr_auc</th>
      <th>test_deviance</th>
      <th>test_brier_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11729</td>
      <td>0.01</td>
      <td>500</td>
      <td>0.803433</td>
      <td>0.185661</td>
      <td>0.174569</td>
      <td>885.786115</td>
      <td>0.040213</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11729</td>
      <td>0.02</td>
      <td>500</td>
      <td>0.794294</td>
      <td>0.186608</td>
      <td>0.175391</td>
      <td>885.608834</td>
      <td>0.040245</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11729</td>
      <td>0.03</td>
      <td>500</td>
      <td>0.790233</td>
      <td>0.179938</td>
      <td>0.168800</td>
      <td>885.986012</td>
      <td>0.040583</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11729</td>
      <td>0.04</td>
      <td>500</td>
      <td>0.773131</td>
      <td>0.172265</td>
      <td>0.161658</td>
      <td>886.580172</td>
      <td>0.041621</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11729</td>
      <td>0.05</td>
      <td>500</td>
      <td>0.763760</td>
      <td>0.169979</td>
      <td>0.163959</td>
      <td>886.331783</td>
      <td>0.043942</td>
    </tr>
  </tbody>
</table>
</div>




```python
ref = metrics_learning[metrics_learning.param_value==0.1].groupby('store_id').mean()
ref.index.name = 'store_id'
ref.reset_index(inplace=True, drop=False)
metrics_learning = metrics_learning[~((metrics_learning.param_value==0.1))]

metrics_learning = pd.concat([ref, metrics_learning], axis=0, sort=False)
metrics_learning.reset_index(inplace=True, drop=True)
print('\033[1mShape of metrics_learning:\033[0m ' + str(metrics_learning.shape) + '.')
metrics_learning.head()
```

    [1mShape of metrics_learning:[0m (1800, 8).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>param_value</th>
      <th>n_estimators</th>
      <th>test_roc_auc</th>
      <th>test_prec_avg</th>
      <th>test_pr_auc</th>
      <th>test_deviance</th>
      <th>test_brier_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>720</td>
      <td>0.1</td>
      <td>300</td>
      <td>0.785362</td>
      <td>0.046540</td>
      <td>0.044608</td>
      <td>1394.933367</td>
      <td>0.015207</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1098</td>
      <td>0.1</td>
      <td>300</td>
      <td>0.946788</td>
      <td>0.605101</td>
      <td>0.600243</td>
      <td>6537.013276</td>
      <td>0.015089</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1181</td>
      <td>0.1</td>
      <td>300</td>
      <td>0.973453</td>
      <td>0.811561</td>
      <td>0.811435</td>
      <td>1180.313850</td>
      <td>0.018021</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1210</td>
      <td>0.1</td>
      <td>300</td>
      <td>0.500000</td>
      <td>0.001986</td>
      <td>0.500993</td>
      <td>1395.998422</td>
      <td>0.001986</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1241</td>
      <td>0.1</td>
      <td>300</td>
      <td>0.857572</td>
      <td>0.781923</td>
      <td>0.777764</td>
      <td>61.084691</td>
      <td>0.231204</td>
    </tr>
  </tbody>
</table>
</div>



<a id='stats_learning_rate'></a>

### Statistics by hyper-parameter value

#### Basic statistics for each performance metric


```python
# Test ROC-AUC:
metrics_learning.groupby('param_value').describe()[['test_roc_auc']].sort_values(('test_roc_auc','mean'),
                                                                                  ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_roc_auc</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.02</th>
      <td>97.0</td>
      <td>0.831906</td>
      <td>0.151156</td>
      <td>0.342522</td>
      <td>0.750781</td>
      <td>0.887913</td>
      <td>0.944998</td>
      <td>0.987302</td>
    </tr>
    <tr>
      <th>0.01</th>
      <td>97.0</td>
      <td>0.831446</td>
      <td>0.140550</td>
      <td>0.480000</td>
      <td>0.758688</td>
      <td>0.875100</td>
      <td>0.939659</td>
      <td>0.978791</td>
    </tr>
    <tr>
      <th>0.03</th>
      <td>97.0</td>
      <td>0.830599</td>
      <td>0.151423</td>
      <td>0.342522</td>
      <td>0.754427</td>
      <td>0.887907</td>
      <td>0.947399</td>
      <td>0.986765</td>
    </tr>
    <tr>
      <th>0.04</th>
      <td>97.0</td>
      <td>0.830495</td>
      <td>0.153136</td>
      <td>0.342522</td>
      <td>0.742089</td>
      <td>0.885976</td>
      <td>0.947927</td>
      <td>0.987979</td>
    </tr>
    <tr>
      <th>0.06</th>
      <td>97.0</td>
      <td>0.829328</td>
      <td>0.153585</td>
      <td>0.342522</td>
      <td>0.775316</td>
      <td>0.888006</td>
      <td>0.948108</td>
      <td>0.987745</td>
    </tr>
    <tr>
      <th>0.09</th>
      <td>97.0</td>
      <td>0.828896</td>
      <td>0.148400</td>
      <td>0.342522</td>
      <td>0.734779</td>
      <td>0.883673</td>
      <td>0.944756</td>
      <td>0.992329</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>97.0</td>
      <td>0.827695</td>
      <td>0.156558</td>
      <td>0.342522</td>
      <td>0.760918</td>
      <td>0.882531</td>
      <td>0.946377</td>
      <td>0.987652</td>
    </tr>
    <tr>
      <th>0.08</th>
      <td>97.0</td>
      <td>0.826583</td>
      <td>0.155345</td>
      <td>0.342522</td>
      <td>0.757088</td>
      <td>0.885916</td>
      <td>0.948056</td>
      <td>0.985142</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>97.0</td>
      <td>0.826562</td>
      <td>0.154379</td>
      <td>0.342522</td>
      <td>0.745703</td>
      <td>0.881290</td>
      <td>0.946419</td>
      <td>0.987465</td>
    </tr>
    <tr>
      <th>0.10</th>
      <td>97.0</td>
      <td>0.823806</td>
      <td>0.150704</td>
      <td>0.342522</td>
      <td>0.766614</td>
      <td>0.874556</td>
      <td>0.941828</td>
      <td>0.992554</td>
    </tr>
    <tr>
      <th>0.20</th>
      <td>97.0</td>
      <td>0.793442</td>
      <td>0.160074</td>
      <td>0.425031</td>
      <td>0.689917</td>
      <td>0.863136</td>
      <td>0.923086</td>
      <td>0.985560</td>
    </tr>
    <tr>
      <th>0.30</th>
      <td>97.0</td>
      <td>0.775078</td>
      <td>0.155401</td>
      <td>0.416717</td>
      <td>0.682255</td>
      <td>0.830538</td>
      <td>0.905835</td>
      <td>0.984162</td>
    </tr>
    <tr>
      <th>0.40</th>
      <td>97.0</td>
      <td>0.740200</td>
      <td>0.164835</td>
      <td>0.100653</td>
      <td>0.614286</td>
      <td>0.754411</td>
      <td>0.862780</td>
      <td>0.980144</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>97.0</td>
      <td>0.712177</td>
      <td>0.159904</td>
      <td>0.080036</td>
      <td>0.592700</td>
      <td>0.720833</td>
      <td>0.845767</td>
      <td>0.946216</td>
    </tr>
    <tr>
      <th>0.60</th>
      <td>97.0</td>
      <td>0.701887</td>
      <td>0.157010</td>
      <td>0.083087</td>
      <td>0.573883</td>
      <td>0.710807</td>
      <td>0.833965</td>
      <td>0.979693</td>
    </tr>
    <tr>
      <th>0.70</th>
      <td>97.0</td>
      <td>0.687094</td>
      <td>0.153518</td>
      <td>0.087964</td>
      <td>0.568805</td>
      <td>0.687981</td>
      <td>0.821167</td>
      <td>0.941187</td>
    </tr>
    <tr>
      <th>0.80</th>
      <td>97.0</td>
      <td>0.674170</td>
      <td>0.150427</td>
      <td>0.077789</td>
      <td>0.564571</td>
      <td>0.694853</td>
      <td>0.790087</td>
      <td>0.931955</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>97.0</td>
      <td>0.648901</td>
      <td>0.151090</td>
      <td>0.077655</td>
      <td>0.538978</td>
      <td>0.643210</td>
      <td>0.753341</td>
      <td>0.963957</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test average precision score:
metrics_learning.groupby('param_value').describe()[['test_prec_avg']].sort_values(('test_prec_avg','mean'),
                                                                                   ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_prec_avg</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.02</th>
      <td>97.0</td>
      <td>0.424598</td>
      <td>0.268898</td>
      <td>0.000955</td>
      <td>0.189846</td>
      <td>0.436735</td>
      <td>0.621557</td>
      <td>0.949442</td>
    </tr>
    <tr>
      <th>0.03</th>
      <td>97.0</td>
      <td>0.423762</td>
      <td>0.271061</td>
      <td>0.000955</td>
      <td>0.197355</td>
      <td>0.425427</td>
      <td>0.645301</td>
      <td>0.950904</td>
    </tr>
    <tr>
      <th>0.04</th>
      <td>97.0</td>
      <td>0.422440</td>
      <td>0.270733</td>
      <td>0.000956</td>
      <td>0.187995</td>
      <td>0.425539</td>
      <td>0.642770</td>
      <td>0.951820</td>
    </tr>
    <tr>
      <th>0.01</th>
      <td>97.0</td>
      <td>0.417226</td>
      <td>0.260441</td>
      <td>0.000955</td>
      <td>0.214814</td>
      <td>0.407508</td>
      <td>0.611690</td>
      <td>0.940401</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>97.0</td>
      <td>0.414802</td>
      <td>0.270464</td>
      <td>0.000955</td>
      <td>0.174480</td>
      <td>0.416324</td>
      <td>0.619688</td>
      <td>0.953089</td>
    </tr>
    <tr>
      <th>0.06</th>
      <td>97.0</td>
      <td>0.414274</td>
      <td>0.270004</td>
      <td>0.000956</td>
      <td>0.179899</td>
      <td>0.436735</td>
      <td>0.601543</td>
      <td>0.952779</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>97.0</td>
      <td>0.404698</td>
      <td>0.270244</td>
      <td>0.000955</td>
      <td>0.180085</td>
      <td>0.382962</td>
      <td>0.611876</td>
      <td>0.951235</td>
    </tr>
    <tr>
      <th>0.08</th>
      <td>97.0</td>
      <td>0.403017</td>
      <td>0.270633</td>
      <td>0.000955</td>
      <td>0.176478</td>
      <td>0.399723</td>
      <td>0.575055</td>
      <td>0.951199</td>
    </tr>
    <tr>
      <th>0.09</th>
      <td>97.0</td>
      <td>0.402995</td>
      <td>0.267075</td>
      <td>0.000955</td>
      <td>0.167407</td>
      <td>0.385198</td>
      <td>0.590707</td>
      <td>0.949985</td>
    </tr>
    <tr>
      <th>0.10</th>
      <td>97.0</td>
      <td>0.389345</td>
      <td>0.268054</td>
      <td>0.000955</td>
      <td>0.153848</td>
      <td>0.377802</td>
      <td>0.575483</td>
      <td>0.951643</td>
    </tr>
    <tr>
      <th>0.20</th>
      <td>97.0</td>
      <td>0.351275</td>
      <td>0.261680</td>
      <td>0.000956</td>
      <td>0.121308</td>
      <td>0.307641</td>
      <td>0.542213</td>
      <td>0.930572</td>
    </tr>
    <tr>
      <th>0.30</th>
      <td>97.0</td>
      <td>0.322726</td>
      <td>0.252101</td>
      <td>0.000956</td>
      <td>0.116799</td>
      <td>0.252266</td>
      <td>0.522759</td>
      <td>0.924380</td>
    </tr>
    <tr>
      <th>0.40</th>
      <td>97.0</td>
      <td>0.280480</td>
      <td>0.227304</td>
      <td>0.000959</td>
      <td>0.094467</td>
      <td>0.236149</td>
      <td>0.443356</td>
      <td>0.932403</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>97.0</td>
      <td>0.265401</td>
      <td>0.219373</td>
      <td>0.000955</td>
      <td>0.083923</td>
      <td>0.195116</td>
      <td>0.436735</td>
      <td>0.922707</td>
    </tr>
    <tr>
      <th>0.70</th>
      <td>97.0</td>
      <td>0.240764</td>
      <td>0.211071</td>
      <td>0.000955</td>
      <td>0.071816</td>
      <td>0.195491</td>
      <td>0.367949</td>
      <td>0.916143</td>
    </tr>
    <tr>
      <th>0.60</th>
      <td>97.0</td>
      <td>0.238675</td>
      <td>0.204233</td>
      <td>0.000955</td>
      <td>0.067817</td>
      <td>0.195381</td>
      <td>0.351546</td>
      <td>0.923208</td>
    </tr>
    <tr>
      <th>0.80</th>
      <td>97.0</td>
      <td>0.224805</td>
      <td>0.200190</td>
      <td>0.000955</td>
      <td>0.064103</td>
      <td>0.167127</td>
      <td>0.326974</td>
      <td>0.919373</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>97.0</td>
      <td>0.207920</td>
      <td>0.200038</td>
      <td>0.000955</td>
      <td>0.051798</td>
      <td>0.156210</td>
      <td>0.313660</td>
      <td>0.911359</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test Brier score:
metrics_learning.groupby('param_value').describe()[['test_brier_score']].sort_values(('test_brier_score','mean'),
                                                                                      ascending=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_brier_score</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.01</th>
      <td>97.0</td>
      <td>0.032972</td>
      <td>0.037547</td>
      <td>0.001986</td>
      <td>0.009247</td>
      <td>0.016245</td>
      <td>0.039237</td>
      <td>0.216399</td>
    </tr>
    <tr>
      <th>0.02</th>
      <td>97.0</td>
      <td>0.033096</td>
      <td>0.037817</td>
      <td>0.001986</td>
      <td>0.009879</td>
      <td>0.017375</td>
      <td>0.039532</td>
      <td>0.216018</td>
    </tr>
    <tr>
      <th>0.03</th>
      <td>97.0</td>
      <td>0.033493</td>
      <td>0.038182</td>
      <td>0.001986</td>
      <td>0.009725</td>
      <td>0.018332</td>
      <td>0.039524</td>
      <td>0.218234</td>
    </tr>
    <tr>
      <th>0.04</th>
      <td>97.0</td>
      <td>0.034058</td>
      <td>0.039239</td>
      <td>0.001938</td>
      <td>0.010107</td>
      <td>0.017868</td>
      <td>0.039527</td>
      <td>0.217495</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>97.0</td>
      <td>0.035292</td>
      <td>0.041008</td>
      <td>0.001986</td>
      <td>0.010289</td>
      <td>0.018188</td>
      <td>0.041328</td>
      <td>0.217613</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>97.0</td>
      <td>0.035815</td>
      <td>0.041279</td>
      <td>0.001986</td>
      <td>0.011298</td>
      <td>0.018436</td>
      <td>0.044369</td>
      <td>0.228494</td>
    </tr>
    <tr>
      <th>0.06</th>
      <td>97.0</td>
      <td>0.035888</td>
      <td>0.041851</td>
      <td>0.001986</td>
      <td>0.010996</td>
      <td>0.018093</td>
      <td>0.039516</td>
      <td>0.226156</td>
    </tr>
    <tr>
      <th>0.10</th>
      <td>97.0</td>
      <td>0.036164</td>
      <td>0.040441</td>
      <td>0.001986</td>
      <td>0.011691</td>
      <td>0.019547</td>
      <td>0.043687</td>
      <td>0.231204</td>
    </tr>
    <tr>
      <th>0.09</th>
      <td>97.0</td>
      <td>0.037640</td>
      <td>0.045146</td>
      <td>0.001986</td>
      <td>0.011562</td>
      <td>0.018715</td>
      <td>0.042488</td>
      <td>0.266379</td>
    </tr>
    <tr>
      <th>0.08</th>
      <td>97.0</td>
      <td>0.038349</td>
      <td>0.049743</td>
      <td>0.001793</td>
      <td>0.011227</td>
      <td>0.018198</td>
      <td>0.041834</td>
      <td>0.306565</td>
    </tr>
    <tr>
      <th>0.20</th>
      <td>97.0</td>
      <td>0.040098</td>
      <td>0.045589</td>
      <td>0.001986</td>
      <td>0.014170</td>
      <td>0.022068</td>
      <td>0.050126</td>
      <td>0.265562</td>
    </tr>
    <tr>
      <th>0.30</th>
      <td>97.0</td>
      <td>0.043414</td>
      <td>0.048165</td>
      <td>0.001986</td>
      <td>0.016387</td>
      <td>0.027274</td>
      <td>0.055107</td>
      <td>0.295489</td>
    </tr>
    <tr>
      <th>0.40</th>
      <td>97.0</td>
      <td>0.049049</td>
      <td>0.059233</td>
      <td>0.001986</td>
      <td>0.016833</td>
      <td>0.028963</td>
      <td>0.060942</td>
      <td>0.340792</td>
    </tr>
    <tr>
      <th>0.70</th>
      <td>97.0</td>
      <td>0.051130</td>
      <td>0.055200</td>
      <td>0.001986</td>
      <td>0.016446</td>
      <td>0.031582</td>
      <td>0.068689</td>
      <td>0.346134</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>97.0</td>
      <td>0.053296</td>
      <td>0.069665</td>
      <td>0.001986</td>
      <td>0.016535</td>
      <td>0.028490</td>
      <td>0.062178</td>
      <td>0.490249</td>
    </tr>
    <tr>
      <th>0.80</th>
      <td>97.0</td>
      <td>0.055023</td>
      <td>0.061641</td>
      <td>0.001986</td>
      <td>0.016013</td>
      <td>0.033118</td>
      <td>0.066027</td>
      <td>0.342762</td>
    </tr>
    <tr>
      <th>0.60</th>
      <td>97.0</td>
      <td>0.055200</td>
      <td>0.070600</td>
      <td>0.001986</td>
      <td>0.017012</td>
      <td>0.031079</td>
      <td>0.058139</td>
      <td>0.382958</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>97.0</td>
      <td>0.059547</td>
      <td>0.073956</td>
      <td>0.001986</td>
      <td>0.017082</td>
      <td>0.035228</td>
      <td>0.075690</td>
      <td>0.524193</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test binomial deviance:
metrics_learning.groupby('param_value').describe()[['test_deviance']].sort_values(('test_deviance','mean'),
                                                                                   ascending=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_deviance</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.07</th>
      <td>97.0</td>
      <td>2427.381294</td>
      <td>3870.917354</td>
      <td>53.383775</td>
      <td>344.973304</td>
      <td>949.500472</td>
      <td>2689.507619</td>
      <td>22106.168159</td>
    </tr>
    <tr>
      <th>0.04</th>
      <td>97.0</td>
      <td>2427.794248</td>
      <td>3870.790707</td>
      <td>53.415953</td>
      <td>344.535102</td>
      <td>949.507244</td>
      <td>2691.750659</td>
      <td>22107.333164</td>
    </tr>
    <tr>
      <th>0.03</th>
      <td>97.0</td>
      <td>2427.831905</td>
      <td>3871.506069</td>
      <td>53.341252</td>
      <td>344.163131</td>
      <td>949.195595</td>
      <td>2697.546202</td>
      <td>22107.456751</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>97.0</td>
      <td>2427.872498</td>
      <td>3870.774418</td>
      <td>53.552137</td>
      <td>344.672183</td>
      <td>949.876339</td>
      <td>2687.297156</td>
      <td>22106.893313</td>
    </tr>
    <tr>
      <th>0.06</th>
      <td>97.0</td>
      <td>2427.919086</td>
      <td>3870.510883</td>
      <td>53.431615</td>
      <td>344.833366</td>
      <td>950.743243</td>
      <td>2693.180159</td>
      <td>22106.014153</td>
    </tr>
    <tr>
      <th>0.09</th>
      <td>97.0</td>
      <td>2427.972148</td>
      <td>3870.527601</td>
      <td>53.186120</td>
      <td>345.239251</td>
      <td>950.393078</td>
      <td>2700.366184</td>
      <td>22108.392434</td>
    </tr>
    <tr>
      <th>0.10</th>
      <td>97.0</td>
      <td>2428.052125</td>
      <td>3872.208850</td>
      <td>53.465944</td>
      <td>344.607794</td>
      <td>950.315976</td>
      <td>2711.822981</td>
      <td>22107.609404</td>
    </tr>
    <tr>
      <th>0.08</th>
      <td>97.0</td>
      <td>2428.199314</td>
      <td>3870.276307</td>
      <td>53.162443</td>
      <td>344.902351</td>
      <td>949.642952</td>
      <td>2692.432251</td>
      <td>22106.965518</td>
    </tr>
    <tr>
      <th>0.02</th>
      <td>97.0</td>
      <td>2428.684011</td>
      <td>3872.098970</td>
      <td>53.516642</td>
      <td>343.723274</td>
      <td>949.064754</td>
      <td>2707.950008</td>
      <td>22108.736274</td>
    </tr>
    <tr>
      <th>0.20</th>
      <td>97.0</td>
      <td>2429.680159</td>
      <td>3872.189548</td>
      <td>53.595908</td>
      <td>343.586806</td>
      <td>950.789731</td>
      <td>2712.755054</td>
      <td>22115.349035</td>
    </tr>
    <tr>
      <th>0.30</th>
      <td>97.0</td>
      <td>2430.699742</td>
      <td>3873.594007</td>
      <td>53.082639</td>
      <td>343.322668</td>
      <td>951.111970</td>
      <td>2719.170824</td>
      <td>22120.687563</td>
    </tr>
    <tr>
      <th>0.01</th>
      <td>97.0</td>
      <td>2430.890670</td>
      <td>3874.249794</td>
      <td>53.395864</td>
      <td>343.333493</td>
      <td>949.287312</td>
      <td>2720.188094</td>
      <td>22113.260987</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>97.0</td>
      <td>2431.451677</td>
      <td>3874.729560</td>
      <td>53.060580</td>
      <td>343.554513</td>
      <td>953.807717</td>
      <td>2720.078857</td>
      <td>22119.997945</td>
    </tr>
    <tr>
      <th>0.60</th>
      <td>97.0</td>
      <td>2432.174312</td>
      <td>3874.101711</td>
      <td>53.061657</td>
      <td>344.767473</td>
      <td>952.516059</td>
      <td>2720.025309</td>
      <td>22115.799983</td>
    </tr>
    <tr>
      <th>0.40</th>
      <td>97.0</td>
      <td>2432.776420</td>
      <td>3874.644853</td>
      <td>53.623527</td>
      <td>344.752205</td>
      <td>953.246668</td>
      <td>2722.032166</td>
      <td>22123.417455</td>
    </tr>
    <tr>
      <th>0.80</th>
      <td>97.0</td>
      <td>2432.802523</td>
      <td>3875.892258</td>
      <td>53.626189</td>
      <td>343.475857</td>
      <td>953.061909</td>
      <td>2730.086679</td>
      <td>22142.724062</td>
    </tr>
    <tr>
      <th>0.70</th>
      <td>97.0</td>
      <td>2433.382861</td>
      <td>3874.475886</td>
      <td>53.626191</td>
      <td>342.129959</td>
      <td>953.169945</td>
      <td>2715.904396</td>
      <td>22119.506149</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>97.0</td>
      <td>2433.672579</td>
      <td>3875.411387</td>
      <td>53.626191</td>
      <td>342.578381</td>
      <td>952.066992</td>
      <td>2726.267200</td>
      <td>22117.007700</td>
    </tr>
  </tbody>
</table>
</div>



#### Averages of performance metrics by hyper-parameter value


```python
metrics_learning.groupby('param_value').mean().sort_values('test_roc_auc', ascending=False).drop('store_id',
                                                                                                 axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_estimators</th>
      <th>test_roc_auc</th>
      <th>test_prec_avg</th>
      <th>test_pr_auc</th>
      <th>test_deviance</th>
      <th>test_brier_score</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.02</th>
      <td>500.0</td>
      <td>0.831906</td>
      <td>0.424598</td>
      <td>0.434306</td>
      <td>2428.684011</td>
      <td>0.033096</td>
    </tr>
    <tr>
      <th>0.01</th>
      <td>500.0</td>
      <td>0.831446</td>
      <td>0.417226</td>
      <td>0.425739</td>
      <td>2430.890670</td>
      <td>0.032972</td>
    </tr>
    <tr>
      <th>0.03</th>
      <td>500.0</td>
      <td>0.830599</td>
      <td>0.423762</td>
      <td>0.433480</td>
      <td>2427.831905</td>
      <td>0.033493</td>
    </tr>
    <tr>
      <th>0.04</th>
      <td>500.0</td>
      <td>0.830495</td>
      <td>0.422440</td>
      <td>0.432729</td>
      <td>2427.794248</td>
      <td>0.034058</td>
    </tr>
    <tr>
      <th>0.06</th>
      <td>500.0</td>
      <td>0.829328</td>
      <td>0.414274</td>
      <td>0.428936</td>
      <td>2427.919086</td>
      <td>0.035888</td>
    </tr>
    <tr>
      <th>0.09</th>
      <td>500.0</td>
      <td>0.828896</td>
      <td>0.402995</td>
      <td>0.418425</td>
      <td>2427.972148</td>
      <td>0.037640</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>500.0</td>
      <td>0.827695</td>
      <td>0.414802</td>
      <td>0.425864</td>
      <td>2427.872498</td>
      <td>0.035292</td>
    </tr>
    <tr>
      <th>0.08</th>
      <td>500.0</td>
      <td>0.826583</td>
      <td>0.403017</td>
      <td>0.419444</td>
      <td>2428.199314</td>
      <td>0.038349</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>500.0</td>
      <td>0.826562</td>
      <td>0.404698</td>
      <td>0.420683</td>
      <td>2427.381294</td>
      <td>0.035815</td>
    </tr>
    <tr>
      <th>0.10</th>
      <td>300.0</td>
      <td>0.823806</td>
      <td>0.389345</td>
      <td>0.405316</td>
      <td>2428.052125</td>
      <td>0.036164</td>
    </tr>
    <tr>
      <th>0.20</th>
      <td>100.0</td>
      <td>0.793442</td>
      <td>0.351275</td>
      <td>0.374188</td>
      <td>2429.680159</td>
      <td>0.040098</td>
    </tr>
    <tr>
      <th>0.30</th>
      <td>100.0</td>
      <td>0.775078</td>
      <td>0.322726</td>
      <td>0.349446</td>
      <td>2430.699742</td>
      <td>0.043414</td>
    </tr>
    <tr>
      <th>0.40</th>
      <td>100.0</td>
      <td>0.740200</td>
      <td>0.280480</td>
      <td>0.323323</td>
      <td>2432.776420</td>
      <td>0.049049</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>100.0</td>
      <td>0.712177</td>
      <td>0.265401</td>
      <td>0.313892</td>
      <td>2431.451677</td>
      <td>0.053296</td>
    </tr>
    <tr>
      <th>0.60</th>
      <td>100.0</td>
      <td>0.701887</td>
      <td>0.238675</td>
      <td>0.297456</td>
      <td>2432.174312</td>
      <td>0.055200</td>
    </tr>
    <tr>
      <th>0.70</th>
      <td>100.0</td>
      <td>0.687094</td>
      <td>0.240764</td>
      <td>0.318514</td>
      <td>2433.382861</td>
      <td>0.051130</td>
    </tr>
    <tr>
      <th>0.80</th>
      <td>100.0</td>
      <td>0.674170</td>
      <td>0.224805</td>
      <td>0.307873</td>
      <td>2432.802523</td>
      <td>0.055023</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>100.0</td>
      <td>0.648901</td>
      <td>0.207920</td>
      <td>0.290774</td>
      <td>2433.672579</td>
      <td>0.059547</td>
    </tr>
  </tbody>
</table>
</div>



#### Frequency of best hyper-parameter values


```python
best_learning_values = metrics_learning.groupby('store_id').idxmax()['test_roc_auc'].values
print('\033[1mRelative frequency of highest performance metric by hyper-parameter value:\033[0m')
print(metrics_learning.reindex(best_learning_values).param_value.value_counts()/len(best_learning_values))
```

    [1mRelative frequency of highest performance metric by hyper-parameter value:[0m
    0.01    0.13
    0.08    0.11
    0.10    0.09
    0.09    0.09
    0.04    0.09
    0.06    0.08
    0.02    0.06
    0.40    0.06
    0.05    0.05
    0.70    0.05
    0.03    0.04
    0.07    0.04
    0.30    0.04
    0.20    0.03
    0.60    0.01
    Name: param_value, dtype: float64
    

#### Average performance metric by best hyper-parameter value


```python
# Dataframe with best hyper-parameter value by dataset:
best_learning_values= metrics_learning.reindex(best_learning_values)[['store_id', 'param_value',
                                                                      'test_roc_auc']]
best_learning_values = best_learning_values.merge(data_info, on='store_id', how='inner')
print('\033[1mShape of best_learning_values:\033[0m ' + str(best_learning_values.shape) + '.')
best_learning_values.head()
```

    [1mShape of best_learning_values:[0m (97, 6).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>param_value</th>
      <th>test_roc_auc</th>
      <th>n_orders</th>
      <th>n_vars</th>
      <th>avg_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>720.0</td>
      <td>0.07</td>
      <td>0.841948</td>
      <td>4028</td>
      <td>1858</td>
      <td>0.011668</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1098.0</td>
      <td>0.04</td>
      <td>0.967290</td>
      <td>19152</td>
      <td>4026</td>
      <td>0.023705</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1181.0</td>
      <td>0.08</td>
      <td>0.981710</td>
      <td>3467</td>
      <td>2698</td>
      <td>0.033458</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1210.0</td>
      <td>0.10</td>
      <td>0.500000</td>
      <td>4028</td>
      <td>2101</td>
      <td>0.001490</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1241.0</td>
      <td>0.30</td>
      <td>0.867388</td>
      <td>206</td>
      <td>3791</td>
      <td>0.320388</td>
    </tr>
  </tbody>
</table>
</div>




```python
best_learning_values.groupby('param_value').mean().sort_values('test_roc_auc', ascending=False)[['test_roc_auc']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test_roc_auc</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.05</th>
      <td>0.949386</td>
    </tr>
    <tr>
      <th>0.08</th>
      <td>0.926989</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>0.914333</td>
    </tr>
    <tr>
      <th>0.09</th>
      <td>0.903461</td>
    </tr>
    <tr>
      <th>0.04</th>
      <td>0.886694</td>
    </tr>
    <tr>
      <th>0.03</th>
      <td>0.874627</td>
    </tr>
    <tr>
      <th>0.01</th>
      <td>0.860046</td>
    </tr>
    <tr>
      <th>0.30</th>
      <td>0.855093</td>
    </tr>
    <tr>
      <th>0.02</th>
      <td>0.852622</td>
    </tr>
    <tr>
      <th>0.06</th>
      <td>0.842446</td>
    </tr>
    <tr>
      <th>0.10</th>
      <td>0.837892</td>
    </tr>
    <tr>
      <th>0.20</th>
      <td>0.806420</td>
    </tr>
    <tr>
      <th>0.40</th>
      <td>0.721395</td>
    </tr>
    <tr>
      <th>0.60</th>
      <td>0.608392</td>
    </tr>
    <tr>
      <th>0.70</th>
      <td>0.545620</td>
    </tr>
  </tbody>
</table>
</div>



<a id='describing_learning_rate_values'></a>

### Describing hyper-parameter values

#### Average numbers of observations and features by best hyper-parameter value


```python
best_learning_values.groupby('param_value').mean()[['n_orders']].sort_values('n_orders', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_orders</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.05</th>
      <td>22486.600000</td>
    </tr>
    <tr>
      <th>0.03</th>
      <td>20053.500000</td>
    </tr>
    <tr>
      <th>0.06</th>
      <td>11759.250000</td>
    </tr>
    <tr>
      <th>0.04</th>
      <td>9694.444444</td>
    </tr>
    <tr>
      <th>0.08</th>
      <td>9280.909091</td>
    </tr>
    <tr>
      <th>0.02</th>
      <td>7018.000000</td>
    </tr>
    <tr>
      <th>0.09</th>
      <td>6210.666667</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>5532.750000</td>
    </tr>
    <tr>
      <th>0.10</th>
      <td>3201.777778</td>
    </tr>
    <tr>
      <th>0.01</th>
      <td>3010.615385</td>
    </tr>
    <tr>
      <th>0.40</th>
      <td>2248.666667</td>
    </tr>
    <tr>
      <th>0.70</th>
      <td>900.400000</td>
    </tr>
    <tr>
      <th>0.30</th>
      <td>796.750000</td>
    </tr>
    <tr>
      <th>0.20</th>
      <td>695.000000</td>
    </tr>
    <tr>
      <th>0.60</th>
      <td>294.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
best_learning_values.groupby('param_value').mean()[['n_vars']].sort_values('n_vars', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_vars</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.30</th>
      <td>2788.500000</td>
    </tr>
    <tr>
      <th>0.04</th>
      <td>2682.555556</td>
    </tr>
    <tr>
      <th>0.70</th>
      <td>2671.000000</td>
    </tr>
    <tr>
      <th>0.06</th>
      <td>2512.875000</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>2353.600000</td>
    </tr>
    <tr>
      <th>0.08</th>
      <td>2346.272727</td>
    </tr>
    <tr>
      <th>0.02</th>
      <td>2340.500000</td>
    </tr>
    <tr>
      <th>0.09</th>
      <td>2312.111111</td>
    </tr>
    <tr>
      <th>0.60</th>
      <td>2279.000000</td>
    </tr>
    <tr>
      <th>0.10</th>
      <td>2252.444444</td>
    </tr>
    <tr>
      <th>0.20</th>
      <td>2245.000000</td>
    </tr>
    <tr>
      <th>0.03</th>
      <td>2213.250000</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>2189.000000</td>
    </tr>
    <tr>
      <th>0.01</th>
      <td>2176.000000</td>
    </tr>
    <tr>
      <th>0.40</th>
      <td>2171.666667</td>
    </tr>
  </tbody>
</table>
</div>



#### Average of response variable by best hyper-parameter value


```python
best_learning_values.groupby('param_value').mean()[['avg_y']].sort_values('avg_y', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_y</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.30</th>
      <td>0.109209</td>
    </tr>
    <tr>
      <th>0.02</th>
      <td>0.086975</td>
    </tr>
    <tr>
      <th>0.40</th>
      <td>0.058605</td>
    </tr>
    <tr>
      <th>0.01</th>
      <td>0.056101</td>
    </tr>
    <tr>
      <th>0.10</th>
      <td>0.051374</td>
    </tr>
    <tr>
      <th>0.06</th>
      <td>0.049796</td>
    </tr>
    <tr>
      <th>0.03</th>
      <td>0.046039</td>
    </tr>
    <tr>
      <th>0.04</th>
      <td>0.045511</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.044855</td>
    </tr>
    <tr>
      <th>0.20</th>
      <td>0.044617</td>
    </tr>
    <tr>
      <th>0.09</th>
      <td>0.040563</td>
    </tr>
    <tr>
      <th>0.60</th>
      <td>0.037415</td>
    </tr>
    <tr>
      <th>0.08</th>
      <td>0.029027</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>0.027704</td>
    </tr>
    <tr>
      <th>0.70</th>
      <td>0.022470</td>
    </tr>
  </tbody>
</table>
</div>



#### Most frequent best hyper-parameter values by quartile of number of observations


```python
best_learning_values['quartile_n_orders'] = percentile_cut(best_learning_values.n_orders, p=4)['percentile']

print('\033[1mFrequency of best hyper-parameter values by quartile of number of observations:\033[0m')
for q in range(1,5):
    print('\033[1mNumber of orders in ' +
          str(np.sort(np.unique(percentile_cut(best_learning_values.n_orders, p=4)['interval']))[q-1]) +
          ' (quartile ' + str(q) + ')\033[0m:')
    print(best_learning_values[best_learning_values.quartile_n_orders==q].param_value.value_counts())
    print('\n')
```

    [1mFrequency of best hyper-parameter values by quartile of number of observations:[0m
    [1mNumber of orders in (156.998, 999.0] (quartile 1)[0m:
    0.70    4
    0.20    3
    0.30    3
    0.01    2
    0.10    2
    0.40    2
    0.08    2
    0.06    2
    0.09    1
    0.03    1
    0.60    1
    0.02    1
    0.04    1
    Name: param_value, dtype: int64
    
    
    [1mNumber of orders in (999.0, 2812.0] (quartile 2)[0m:
    0.01    7
    0.09    4
    0.10    3
    0.40    3
    0.08    2
    0.04    1
    0.70    1
    0.06    1
    0.02    1
    0.30    1
    Name: param_value, dtype: int64
    
    
    [1mNumber of orders in (2812.0, 7946.0] (quartile 3)[0m:
    0.10    4
    0.08    4
    0.01    3
    0.07    3
    0.04    2
    0.09    2
    0.06    2
    0.02    2
    0.05    1
    0.03    1
    Name: param_value, dtype: int64
    
    
    [1mNumber of orders in (7946.0, 63963.001] (quartile 4)[0m:
    0.04    5
    0.05    4
    0.08    3
    0.06    3
    0.09    2
    0.02    2
    0.03    2
    0.40    1
    0.01    1
    0.07    1
    Name: param_value, dtype: int64
    
    
    

#### Most frequent best hyper-parameter values by quartile of number of features


```python
best_learning_values['quartile_n_vars'] = percentile_cut(best_learning_values.n_vars, p=4)['percentile']

print('\033[1mFrequency of best hyper-parameter values by quartile of number of features:\033[0m')
for q in range(1,5):
    print('\033[1mNumber of vars in ' +
          str(np.sort(np.unique(percentile_cut(best_learning_values.n_vars, p=4)['interval']))[q-1]) +
          ' (quartile ' + str(q) + ')\033[0m:')
    print(best_learning_values[best_learning_values.quartile_n_vars==q].param_value.value_counts())
    print('\n')
```

    [1mFrequency of best hyper-parameter values by quartile of number of features:[0m
    [1mNumber of vars in (1415.998, 2069.0] (quartile 1)[0m:
    0.01    6
    0.40    3
    0.08    3
    0.09    2
    0.20    2
    0.03    2
    0.07    2
    0.10    2
    0.04    1
    0.70    1
    0.02    1
    Name: param_value, dtype: int64
    
    
    [1mNumber of vars in (2069.0, 2321.0] (quartile 2)[0m:
    0.06    5
    0.09    4
    0.10    3
    0.08    3
    0.05    2
    0.02    2
    0.01    1
    0.70    1
    0.07    1
    0.60    1
    0.03    1
    Name: param_value, dtype: int64
    
    
    [1mNumber of vars in (2321.0, 2534.0] (quartile 3)[0m:
    0.04    5
    0.10    4
    0.30    3
    0.08    2
    0.09    2
    0.40    2
    0.01    2
    0.02    2
    0.05    1
    0.06    1
    Name: param_value, dtype: int64
    
    
    [1mNumber of vars in (2534.0, 4026.001] (quartile 4)[0m:
    0.01    4
    0.70    3
    0.04    3
    0.08    3
    0.05    2
    0.06    2
    0.02    1
    0.09    1
    0.20    1
    0.07    1
    0.40    1
    0.30    1
    0.03    1
    Name: param_value, dtype: int64
    
    
    

#### Correlation between performance metric of best hyper-parameter value and dataset information


```python
best_learning_values[['test_roc_auc', 'n_orders', 'n_vars', 'avg_y']].corr()[['test_roc_auc']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test_roc_auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>test_roc_auc</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>n_orders</th>
      <td>0.325385</td>
    </tr>
    <tr>
      <th>n_vars</th>
      <td>0.068162</td>
    </tr>
    <tr>
      <th>avg_y</th>
      <td>0.146538</td>
    </tr>
  </tbody>
</table>
</div>



#### Correlation between performance metric and dataset information by hyper-parameter value


```python
metrics_learning = metrics_learning.merge(data_info, on='store_id', how='left')
print('\033[1mShape of metrics_learning:\033[0m ' + str(metrics_learning.shape) + '.')
metrics_learning.head()
```

    [1mShape of metrics_learning:[0m (1800, 11).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>param_value</th>
      <th>n_estimators</th>
      <th>test_roc_auc</th>
      <th>test_prec_avg</th>
      <th>test_pr_auc</th>
      <th>test_deviance</th>
      <th>test_brier_score</th>
      <th>n_orders</th>
      <th>n_vars</th>
      <th>avg_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>720</td>
      <td>0.1</td>
      <td>300</td>
      <td>0.785362</td>
      <td>0.046540</td>
      <td>0.044608</td>
      <td>1394.933367</td>
      <td>0.015207</td>
      <td>4028</td>
      <td>1858</td>
      <td>0.011668</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1098</td>
      <td>0.1</td>
      <td>300</td>
      <td>0.946788</td>
      <td>0.605101</td>
      <td>0.600243</td>
      <td>6537.013276</td>
      <td>0.015089</td>
      <td>19152</td>
      <td>4026</td>
      <td>0.023705</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1181</td>
      <td>0.1</td>
      <td>300</td>
      <td>0.973453</td>
      <td>0.811561</td>
      <td>0.811435</td>
      <td>1180.313850</td>
      <td>0.018021</td>
      <td>3467</td>
      <td>2698</td>
      <td>0.033458</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1210</td>
      <td>0.1</td>
      <td>300</td>
      <td>0.500000</td>
      <td>0.001986</td>
      <td>0.500993</td>
      <td>1395.998422</td>
      <td>0.001986</td>
      <td>4028</td>
      <td>2101</td>
      <td>0.001490</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1241</td>
      <td>0.1</td>
      <td>300</td>
      <td>0.857572</td>
      <td>0.781923</td>
      <td>0.777764</td>
      <td>61.084691</td>
      <td>0.231204</td>
      <td>206</td>
      <td>3791</td>
      <td>0.320388</td>
    </tr>
  </tbody>
</table>
</div>




```python
for v in np.sort(metrics_learning.param_value.unique()):
    print('\033[1mlearning_rate = ' + str(v) + '\033[0m')
    print(metrics_learning[metrics_learning.param_value==v][['test_roc_auc',
                                                               'n_orders', 'n_vars',
                                                               'avg_y']].corr()[['test_roc_auc']])
    print('\n')
```

    [1mlearning_rate = 0.01[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.320640
    n_vars            0.054209
    avg_y             0.168115
    
    
    [1mlearning_rate = 0.02[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.325343
    n_vars            0.051487
    avg_y             0.160536
    
    
    [1mlearning_rate = 0.03[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.315924
    n_vars            0.049012
    avg_y             0.165841
    
    
    [1mlearning_rate = 0.04[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.310994
    n_vars            0.050265
    avg_y             0.166318
    
    
    [1mlearning_rate = 0.05[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.305277
    n_vars            0.053582
    avg_y             0.166577
    
    
    [1mlearning_rate = 0.06[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.305577
    n_vars            0.053258
    avg_y             0.162162
    
    
    [1mlearning_rate = 0.07[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.308657
    n_vars            0.058568
    avg_y             0.168627
    
    
    [1mlearning_rate = 0.08[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.300835
    n_vars            0.050071
    avg_y             0.165390
    
    
    [1mlearning_rate = 0.09[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.321595
    n_vars            0.067840
    avg_y             0.160407
    
    
    [1mlearning_rate = 0.1[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.307627
    n_vars            0.076420
    avg_y             0.181971
    
    
    [1mlearning_rate = 0.2[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.231956
    n_vars            0.085740
    avg_y             0.264500
    
    
    [1mlearning_rate = 0.3[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.187938
    n_vars            0.134440
    avg_y             0.262003
    
    
    [1mlearning_rate = 0.4[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.137671
    n_vars            0.143738
    avg_y             0.215016
    
    
    [1mlearning_rate = 0.5[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.157379
    n_vars            0.164513
    avg_y             0.264487
    
    
    [1mlearning_rate = 0.6[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.181419
    n_vars            0.164364
    avg_y             0.257878
    
    
    [1mlearning_rate = 0.7[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.171492
    n_vars            0.214857
    avg_y             0.283341
    
    
    [1mlearning_rate = 0.8[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.085145
    n_vars            0.250912
    avg_y             0.297362
    
    
    [1mlearning_rate = 0.9[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.125291
    n_vars            0.121289
    avg_y             0.349575
    
    
    

<a id='data_vis_learning_rate'></a>

### Data visualization

<a id='reference16'></a>
#### Average of performance metric by hyper-parameter value


```python
# Select a performance metric:
metric = 'test_roc_auc'

fig=px.scatter(x=np.sort(metrics_learning['param_value'].apply(lambda x: 'v = ' + str(x)).unique()),
               y=metrics_learning.sort_values('param_value').groupby('param_value').mean()[metric], 
               error_y=np.array(metrics_learning.groupby('param_value').std()['test_roc_auc']),
               color_discrete_sequence=['#0b6fab'],
               width=900, height=500,
               title='Average of ' + metric + ' by learning rate value',
               labels={'y': metric, 'x': ''})

fig.add_trace(
    go.Scatter(
        x=np.sort(metrics_learning['param_value'].apply(lambda x: 'v = ' + str(x)).unique()),
        y=metrics_learning.sort_values('param_value').groupby('param_value').mean()[metric],
        line = dict(color='#0b6fab', width=2, dash='dash'),
        name='avg_' + metric
              )
)
```


<div>                            <div id="31eb95fd-98ee-4479-9f07-ee40c5473ffe" class="plotly-graph-div" style="height:500px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("31eb95fd-98ee-4479-9f07-ee40c5473ffe")) {                    Plotly.newPlot(                        "31eb95fd-98ee-4479-9f07-ee40c5473ffe",                        [{"error_y": {"array": [0.1405500435249385, 0.151156223888752, 0.1514227076522514, 0.15313576401514187, 0.15655820096186573, 0.15358466937409113, 0.15437912139980267, 0.1553448925136026, 0.1483998939023502, 0.15070379395549757, 0.1600744174732837, 0.15540128644364584, 0.16483541716103797, 0.1599037029226715, 0.15701018060934363, 0.15351783477000336, 0.15042722706467365, 0.15109027619438217]}, "hovertemplate": "=%{x}<br>test_roc_auc=%{y}<extra></extra>", "legendgroup": "", "marker": {"color": "#0b6fab", "symbol": "circle"}, "mode": "markers", "name": "", "orientation": "v", "showlegend": false, "type": "scatter", "x": ["v = 0.01", "v = 0.02", "v = 0.03", "v = 0.04", "v = 0.05", "v = 0.06", "v = 0.07", "v = 0.08", "v = 0.09", "v = 0.1", "v = 0.2", "v = 0.3", "v = 0.4", "v = 0.5", "v = 0.6", "v = 0.7", "v = 0.8", "v = 0.9"], "xaxis": "x", "y": [0.8314458889283353, 0.8319055635370757, 0.8305991244667186, 0.8304946026149943, 0.8276950791598751, 0.8293275181649004, 0.8265622867100382, 0.8265832703352288, 0.8288962970120715, 0.8238064682775488, 0.793442451916717, 0.7750778313197179, 0.7401998632752808, 0.7121772439566479, 0.7018871073139623, 0.6870939288826063, 0.6741703105051983, 0.648900582490439], "yaxis": "y"}, {"line": {"color": "#0b6fab", "dash": "dash", "width": 2}, "name": "avg_test_roc_auc", "type": "scatter", "x": ["v = 0.01", "v = 0.02", "v = 0.03", "v = 0.04", "v = 0.05", "v = 0.06", "v = 0.07", "v = 0.08", "v = 0.09", "v = 0.1", "v = 0.2", "v = 0.3", "v = 0.4", "v = 0.5", "v = 0.6", "v = 0.7", "v = 0.8", "v = 0.9"], "y": [0.8314458889283353, 0.8319055635370757, 0.8305991244667186, 0.8304946026149943, 0.8276950791598751, 0.8293275181649004, 0.8265622867100382, 0.8265832703352288, 0.8288962970120715, 0.8238064682775488, 0.793442451916717, 0.7750778313197179, 0.7401998632752808, 0.7121772439566479, 0.7018871073139623, 0.6870939288826063, 0.6741703105051983, 0.648900582490439]}],                        {"height": 500, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Average of test_roc_auc by learning rate value"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": ""}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "test_roc_auc"}}},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('31eb95fd-98ee-4479-9f07-ee40c5473ffe');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

<a id='reference18'></a>
#### Boxplot of performance metric by hyper-parameter value


```python
# Select a performance metric:
metric = 'test_roc_auc'

px.box(data_frame=metrics_learning.sort_values('param_value'),
       x=np.sort(metrics_learning['param_value'].apply(lambda x: 'v = ' + str(x))),
       y=metric, hover_data=['store_id'],
       color_discrete_sequence=['#0b6fab'],
       width=900, height=500,
       labels={'x': ' '},
       title='Distribution of ' + metric + ' by learning rate value')
```


<div>                            <div id="6909a156-60bc-4403-81b6-b3dd83cbf1a7" class="plotly-graph-div" style="height:500px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("6909a156-60bc-4403-81b6-b3dd83cbf1a7")) {                    Plotly.newPlot(                        "6909a156-60bc-4403-81b6-b3dd83cbf1a7",                        [{"alignmentgroup": "True", "customdata": [[720], [3146], [7292], [3962], [6044], [8181], [7333], [2352], [6970], [9491], [5847], [5939], [8790], [7939], [6078], [7630], [10268], [1098], [8358], [10060], [4408], [5342], [6105], [4268], [11729], [10311], [1424], [7988], [4736], [7755], [3481], [6083], [4838], [8446], [5848], [6106], [5860], [1559], [10650], [3781], [6256], [8436], [5428], [11223], [6170], [5168], [5663], [2866], [3437], [4636], [6929], [7790], [9098], [8894], [9177], [10349], [1181], [1739], [3988], [1549], [8421], [4601], [7849], [5593], [6714], [5085], [8783], [6047], [9761], [8832], [7062], [1961], [9409], [7845], [8282], [1241], [2699], [6004], [5215], [2868], [1603], [1875], [2212], [9281], [9541], [7161], [5394], [1210], [7185], [12658], [11723], [2782], [4974], [4030], [1979], [6971], [2056], [3859], [12980], [6966], [1961], [7062], [8358], [2212], [5394], [3781], [7845], [5215], [5342], [2699], [4408], [2782], [3962], [8446], [11723], [6004], [1559], [5860], [10650], [2868], [1603], [1241], [6106], [7185], [8832], [7292], [3146], [6078], [5847], [5939], [720], [7630], [10268], [9491], [1098], [1210], [6970], [6971], [8790], [10060], [2352], [6256], [4030], [9281], [8181], [8436], [7333], [5428], [6714], [5085], [6044], [7161], [8783], [9761], [1875], [6047], [2056], [5593], [7939], [6929], [4268], [5663], [10311], [7755], [2866], [1979], [9098], [11729], [4636], [4736], [3437], [12980], [10349], [7790], [1739], [7988], [1181], [8282], [6966], [1424], [8894], [3481], [3988], [5168], [6170], [4601], [9541], [5848], [7849], [3859], [8421], [4974], [9409], [11223], [9177], [4838], [12658], [6105], [6083], [1549], [6971], [8436], [8282], [9177], [7630], [4030], [12980], [1181], [10268], [9098], [9541], [6714], [6256], [6078], [10349], [8790], [3988], [1549], [720], [1098], [1739], [10060], [5593], [5085], [5663], [6170], [2699], [5215], [4974], [11223], [6004], [1603], [7939], [2782], [2868], [7849], [1241], [8421], [1875], [7185], [8894], [7845], [2212], [6966], [8783], [9761], [6929], [4601], [3859], [6047], [7790], [7062], [2056], [3437], [8832], [4636], [2866], [1961], [5168], [6105], [5428], [6083], [8446], [10311], [2352], [6970], [5860], [6106], [7161], [5342], [10650], [4268], [4408], [11723], [1559], [8181], [1979], [11729], [7333], [4736], [6044], [7988], [1210], [7755], [5847], [12658], [4838], [3962], [9281], [5394], [9409], [8358], [3146], [3481], [5939], [5848], [1424], [7292], [3781], [9491], [5215], [2056], [3146], [7755], [3437], [8358], [6929], [1961], [5342], [8282], [7185], [2212], [3859], [3781], [7845], [2699], [7988], [4736], [1559], [3481], [6170], [11223], [4838], [5663], [1241], [7849], [4974], [12658], [8421], [5848], [6083], [9281], [5593], [10650], [4601], [4636], [1875], [5860], [2866], [6106], [2868], [7790], [2782], [1603], [6004], [5168], [10311], [9098], [8832], [5847], [11729], [10349], [7333], [1210], [8436], [6714], [8181], [5428], [6256], [4268], [12980], [6971], [1098], [5939], [3988], [4030], [6970], [6105], [10060], [2352], [9491], [11723], [10268], [6078], [5085], [6044], [7630], [4408], [8894], [7062], [7161], [720], [8446], [6047], [5394], [7939], [1181], [1549], [7292], [9541], [9409], [9761], [6966], [8783], [1979], [9177], [8790], [3962], [1739], [1424], [9281], [4636], [1549], [6170], [9098], [6105], [9409], [10311], [4974], [1979], [8894], [1181], [7755], [6929], [1739], [9177], [7988], [3437], [1424], [5168], [6966], [10349], [3481], [4268], [11723], [7790], [2866], [3988], [4736], [6083], [9541], [11729], [3859], [2868], [8282], [5085], [8790], [9761], [7333], [8446], [8783], [7292], [7062], [5394], [6047], [4408], [8832], [5663], [2056], [1961], [4030], [6044], [7161], [8436], [1210], [7939], [1098], [5847], [6078], [9491], [10268], [2212], [5939], [10060], [5428], [6714], [8181], [6256], [6971], [6970], [2352], [3781], [12980], [7845], [12658], [7185], [3146], [4601], [8421], [10650], [2782], [5593], [5860], [7849], [720], [4838], [11223], [1241], [5848], [6106], [1875], [7630], [6004], [1559], [8358], [3962], [1603], [2699], [5215], [5342], [5939], [2212], [10650], [8358], [9541], [1098], [5663], [1181], [11723], [7333], [5428], [9281], [7185], [8446], [1241], [7790], [1603], [720], [4601], [7161], [1424], [9761], [6083], [8790], [12980], [1549], [8436], [7988], [5085], [7755], [10311], [8783], [7062], [6047], [6256], [11729], [8832], [1961], [3781], [7845], [1979], [5215], [2699], [6105], [4268], [6714], [10060], [4736], [5342], [4408], [7292], [1559], [6044], [6970], [8181], [6106], [5848], [2352], [5860], [9491], [5847], [4838], [7630], [7939], [6078], [3481], [10268], [6004], [12658], [9409], [3146], [6929], [3962], [3437], [2866], [2868], [1739], [8894], [2056], [9177], [4030], [5168], [4636], [4974], [6170], [6971], [10349], [11223], [8282], [9098], [3859], [1875], [5394], [8421], [3988], [5593], [2782], [7849], [1210], [6966], [1559], [720], [5848], [8358], [10650], [4838], [2782], [4974], [7185], [6106], [3146], [6966], [9281], [3481], [12658], [6105], [1979], [9409], [4268], [5394], [1424], [1210], [11729], [6971], [7755], [10311], [7161], [11723], [12980], [6083], [7988], [4736], [4030], [5860], [2056], [5847], [3781], [1603], [2699], [6004], [2868], [8421], [4601], [1875], [5593], [7849], [5663], [11223], [4636], [6170], [7845], [5168], [7790], [3859], [2866], [3437], [6929], [1739], [1181], [8894], [9177], [10349], [9541], [3988], [9098], [1241], [5215], [3962], [1961], [4408], [7333], [7292], [6970], [6044], [8181], [8446], [5939], [2352], [9491], [7630], [1549], [1098], [7939], [6078], [10268], [6714], [5428], [10060], [6256], [8436], [5085], [9761], [8282], [7062], [8783], [6047], [8790], [2212], [5342], [8832], [3859], [11723], [4030], [6105], [7755], [5860], [3146], [6970], [7630], [720], [6714], [8421], [4636], [1739], [9098], [1241], [8790], [8446], [7062], [5215], [5168], [3962], [1875], [5593], [7849], [6106], [11223], [5848], [6170], [4838], [2866], [3437], [3481], [6929], [8894], [4736], [10349], [7988], [3988], [10311], [1549], [1559], [2868], [5342], [6004], [7939], [9491], [6078], [10268], [2352], [10060], [6256], [8181], [8436], [9541], [6044], [8783], [7292], [6047], [8832], [4408], [1961], [7845], [3781], [2699], [5085], [11729], [9177], [5847], [6083], [7161], [5428], [2782], [1098], [9281], [5394], [5939], [1210], [1181], [7333], [1424], [8358], [6971], [10650], [12658], [2212], [9761], [2056], [1979], [7185], [1603], [4268], [12980], [4601], [7790], [4974], [8282], [9409], [5663], [6966], [6970], [6966], [2056], [8181], [7292], [2352], [4974], [6044], [2782], [7630], [4030], [7755], [6971], [11729], [4268], [10311], [7988], [9491], [9409], [4736], [5860], [1979], [4408], [3481], [12658], [5848], [6105], [6106], [3146], [5394], [1210], [5342], [3781], [4838], [1559], [5215], [7939], [3988], [1549], [9541], [1241], [1181], [7790], [5663], [4601], [8790], [1603], [2212], [9761], [5428], [10349], [8446], [5939], [7333], [8358], [11723], [10650], [6083], [1424], [9281], [720], [7161], [7185], [12980], [8282], [1098], [9098], [9177], [8894], [6078], [6714], [10268], [10060], [6256], [8436], [7062], [5085], [8783], [6047], [8832], [3962], [1961], [7845], [2699], [6004], [8421], [2868], [1875], [5593], [7849], [4636], [11223], [6170], [5168], [2866], [1739], [3437], [6929], [5847], [3859], [720], [12980], [4030], [4268], [4408], [4601], [4636], [4736], [4838], [4974], [5085], [5168], [5215], [5342], [5394], [5428], [5593], [5663], [5847], [5848], [5860], [5939], [6004], [3988], [6044], [3962], [3781], [1098], [1181], [1210], [1241], [1424], [1549], [1559], [1603], [1875], [1961], [1979], [2056], [2212], [2352], [2699], [2782], [2866], [2868], [3146], [3437], [3481], [3859], [6047], [1739], [6083], [8446], [8783], [8790], [8832], [8894], [9098], [9177], [9281], [9409], [9491], [9541], [9761], [10060], [10268], [10311], [10349], [11223], [11723], [11729], [12658], [6078], [8436], [8421], [10650], [8282], [6105], [8358], [6106], [6170], [6714], [6929], [6966], [6970], [6971], [7062], [7161], [6256], [7292], [7185], [7939], [7845], [7790], [7849], [7630], [8181], [7333], [7755], [7988], [8790], [8832], [6004], [12658], [1961], [7845], [2699], [1979], [5593], [2868], [1875], [9409], [7849], [11223], [8783], [6170], [11723], [8446], [5085], [9098], [8436], [7292], [4268], [6044], [2056], [8181], [2352], [9491], [6971], [5847], [7939], [6078], [10268], [1210], [10060], [1241], [7062], [6256], [5394], [5168], [9761], [2866], [5428], [5860], [1098], [12980], [7630], [5939], [7333], [3146], [8358], [10650], [6083], [7185], [1424], [6970], [9281], [7161], [1739], [2212], [6714], [8282], [7755], [3437], [6929], [3859], [720], [8894], [9177], [10349], [4030], [3988], [1549], [3962], [9541], [1181], [6105], [7790], [5663], [4601], [1603], [4408], [6047], [8421], [5342], [4838], [5215], [1559], [7988], [3481], [6106], [2782], [10311], [4636], [6966], [5848], [4974], [4736], [11729], [3781], [8436], [4736], [5593], [4974], [1181], [9409], [7062], [5428], [12980], [1098], [6256], [8421], [1559], [10060], [7849], [1241], [5939], [9541], [10268], [4601], [1210], [11223], [6105], [5394], [1875], [5085], [4636], [2699], [7845], [8282], [7755], [1603], [1979], [1961], [6004], [7988], [8832], [6106], [5215], [12658], [8446], [2212], [6966], [6047], [5848], [2868], [8790], [9761], [8783], [7790], [7630], [5860], [5663], [11723], [6714], [6078], [11729], [9491], [7292], [10650], [720], [3988], [2866], [9177], [2352], [7185], [4030], [6083], [10349], [6929], [3437], [2056], [8181], [2782], [6970], [6044], [1424], [9281], [5168], [3962], [6971], [7161], [8894], [10311], [7333], [6170], [3146], [1739], [3781], [4408], [9098], [7939], [8358], [3481], [3859], [5342], [5847], [4838], [4268], [1549], [9177], [5848], [8282], [6004], [8446], [3481], [6929], [5663], [7755], [6714], [2699], [1875], [720], [1549], [11223], [6170], [3781], [11723], [9541], [7849], [4636], [6105], [4974], [4838], [3859], [5168], [3962], [3988], [9409], [2866], [1181], [4268], [10349], [4030], [7790], [3437], [2868], [5593], [4736], [2212], [4601], [7185], [5847], [2782], [8358], [7939], [7333], [3146], [6078], [1979], [5939], [10268], [10311], [1241], [1559], [10060], [8421], [1098], [9491], [6971], [1739], [4408], [9281], [7292], [11729], [7630], [2056], [6044], [1424], [6970], [8181], [6083], [2352], [10650], [9098], [5342], [6256], [1210], [12980], [6047], [7161], [12658], [8832], [8783], [7988], [9761], [8790], [7062], [8894], [5085], [1961], [5860], [6106], [5394], [6966], [5428], [5215], [8436], [1603], [7845], [4601], [6970], [10349], [8421], [6083], [1603], [4030], [7755], [9177], [11729], [7630], [1424], [5663], [9281], [7988], [2212], [3988], [1098], [10311], [6714], [9541], [5428], [1181], [5939], [6105], [4736], [3146], [1549], [12980], [7333], [9761], [8358], [7185], [7790], [8282], [5860], [6966], [3962], [10650], [3781], [6170], [8894], [8436], [5085], [8790], [6106], [8783], [12658], [6047], [8832], [5394], [7062], [7845], [8446], [1979], [2699], [5848], [6004], [2868], [9409], [1961], [6256], [10060], [1559], [1739], [4408], [2056], [7292], [5215], [6044], [8181], [9098], [5342], [2352], [6971], [9491], [5847], [2782], [7939], [1210], [6078], [1241], [10268], [1875], [5593], [7161], [4636], [3437], [4268], [4838], [11223], [720], [11723], [5168], [7849], [6929], [3859], [2866], [3481], [4974], [5939], [3859], [7185], [8358], [1098], [5847], [10311], [4974], [6966], [10349], [1559], [7333], [7939], [1241], [9491], [10268], [3962], [2866], [6078], [3146], [1210], [10060], [2352], [3781], [7630], [7161], [9281], [4030], [4408], [2056], [6929], [7292], [3481], [1424], [3437], [6044], [5215], [1739], [5342], [11729], [6083], [9177], [8181], [9098], [6971], [6970], [10650], [720], [4636], [5593], [2212], [2782], [6105], [5394], [7988], [8446], [1979], [4601], [7845], [4838], [4268], [2699], [5848], [5663], [11223], [7849], [6004], [7062], [7755], [8282], [7790], [2868], [8421], [9409], [11723], [6714], [1181], [1875], [9541], [1603], [1961], [9761], [5168], [5085], [8790], [8436], [8832], [4736], [6170], [5428], [8783], [12658], [6106], [5860], [12980], [8894], [3988], [6047], [6256], [1549], [7161], [11723], [720], [8790], [8282], [6714], [1739], [9098], [7185], [3962], [1241], [5215], [8446], [12980], [4636], [7062], [2868], [9281], [7939], [6078], [10268], [10060], [5394], [6256], [8436], [5085], [12658], [8783], [6047], [8832], [1961], [1979], [7845], [2699], [7630], [9409], [1875], [1210], [5847], [9491], [2352], [11729], [6966], [10311], [7988], [4736], [4974], [3481], [4838], [5848], [5593], [6106], [1559], [5342], [3781], [2056], [4408], [7292], [6044], [8181], [6971], [2782], [7849], [6004], [7790], [1181], [4268], [7755], [5663], [4601], [1603], [5860], [2212], [9761], [5428], [1098], [3146], [5939], [7333], [8358], [6970], [10650], [6083], [1424], [9541], [1549], [8421], [6105], [3859], [6929], [5168], [8894], [3437], [4030], [2866], [10349], [3988], [6170], [11223], [9177], [8421], [4974], [6970], [7062], [3859], [7630], [2782], [9409], [1979], [5394], [2056], [4030], [6714], [5860], [6971], [12658], [1210], [5215], [7755], [6105], [3146], [4268], [9177], [1603], [7939], [10349], [5847], [3988], [1549], [9541], [9491], [1181], [2352], [7790], [5663], [8181], [4601], [1961], [6044], [2212], [9761], [7292], [5428], [6966], [4408], [8894], [6929], [7845], [6047], [2699], [8783], [6004], [2868], [5085], [1875], [5593], [8436], [7849], [11223], [6256], [6170], [10060], [5168], [2866], [3437], [10268], [6078], [1098], [8832], [5939], [5848], [12980], [8282], [3962], [3481], [720], [4736], [11723], [8446], [8790], [7988], [1241], [10311], [9098], [1739], [11729], [4636], [7185], [7161], [4838], [10650], [6106], [3781], [7333], [5342], [8358], [1424], [6083], [9281], [1559], [6256], [7988], [8436], [12658], [5848], [7292], [5085], [2056], [11729], [8783], [4408], [6047], [10311], [5342], [10060], [6971], [9491], [1210], [2782], [2352], [5847], [6106], [4838], [7939], [3781], [8181], [6078], [3481], [5394], [1559], [10268], [4736], [6044], [4974], [8421], [3859], [8832], [9761], [3146], [5428], [1098], [5939], [7333], [6970], [8358], [10650], [6083], [7630], [1424], [9281], [2212], [7161], [6714], [12980], [8282], [3962], [7062], [720], [11723], [8446], [8790], [5215], [1241], [9098], [1739], [7185], [1603], [5860], [4601], [1961], [7845], [2699], [9409], [6004], [2868], [1875], [4268], [5593], [7849], [11223], [6170], [4636], [5168], [2866], [3437], [4030], [6929], [8894], [9177], [10349], [6105], [3988], [1549], [9541], [7755], [1181], [7790], [5663], [1979], [6966]], "hovertemplate": " =%{x}<br>test_roc_auc=%{y}<br>store_id=%{customdata[0]}<extra></extra>", "legendgroup": "", "marker": {"color": "#0b6fab"}, "name": "", "notched": false, "offsetgroup": "", "orientation": "v", "showlegend": false, "type": "box", "x": ["v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.5", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.6", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.8", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9", "v = 0.9"], "x0": " ", "xaxis": "x", "y": [0.8070307281229123, 0.7661392405063291, 0.49410377358490565, 0.9551789799284467, 0.9589292762500923, 0.4970667870036101, 0.9432092542863044, 0.8600296007893543, 0.8988186885223858, 0.9427519981599678, 0.9603873660933875, 0.9787906137184116, 0.8122947454844006, 0.9324832775919732, 0.48, 0.6158730158730159, 0.5379704301075269, 0.9488329220673627, 0.8836542861200082, 0.8672393802044902, 0.7252604166666665, 0.9290982544315352, 0.9605073238855135, 0.8618624735212644, 0.8034333953272115, 0.8478452620967742, 0.9489703726769274, 0.5706841138659321, 0.8896799116997793, 0.8654431216931217, 0.7586877526997288, 0.793440736478711, 0.9664097153563649, 0.48852772466539196, 0.9698025551684089, 0.7131026323572908, null, 0.9150462962962964, 0.8751003693592418, 0.9595667659251474, 0.9224583075149745, 0.6591090957588691, 0.9096824850492623, 0.732959953144447, 0.9739443246391161, 0.5568181818181819, 0.9374862624758847, 0.9396589541292537, 0.9423510466988728, 0.6244318181818181, 0.8918216175359032, 0.7036927330173776, 0.854729428545957, 0.8550540368722187, 0.928524502584539, 0.8005857407673881, 0.974511863568957, 0.7332572298325724, 0.9773926237161532, null, 0.4995625546806649, 0.9749213669776647, 0.8694032327657738, 0.9305811664255725, 0.7177038817663819, 0.7402380952380953, 0.9040342792465034, 0.9290654238684186, 0.7864237348538845, 0.9439868204283361, null, 0.9171615761689291, 0.9586450927779896, 0.7303921568627452, 0.9075447741245656, 0.8513621794871795, 0.9714348552828364, 0.8570744846286942, 0.8894950791613179, 0.951559633027523, 0.9668197194088163, 0.9534917840375587, 0.899432636887608, 0.8802118548568852, 0.9242022299115724, 0.7883373688458435, 0.8705131028407841, 0.5, 0.7478316187993608, 0.8031249999999999, 0.905123743232792, 0.6831283817019183, 0.8622448979591837, 0.8479006249404731, 0.6921953339802686, 0.4809384164222874, 0.9366336431833007, 0.9579514624603415, 0.9035469107551487, 0.8143083436133229, 0.9213565233785822, null, 0.8916498366966976, 0.9344650576368876, 0.8683109447258314, 0.9645560354710222, 0.729983660130719, 0.8879129788527801, 0.9355779307857874, 0.9755212457112695, 0.7507812500000001, 0.6841613379242498, 0.9647733371931823, 0.48996175908221795, 0.9024506960556845, 0.8810984723811709, 0.9347993827160493, null, 0.8674053851506878, 0.9742966360856269, 0.9716603282380302, 0.8565705128205128, 0.7131026323572908, 0.7482621918105788, 0.9375280814737157, 0.49410377358490565, 0.7401898734177215, 0.48, 0.9692662287632057, 0.9796931407942238, 0.8387886884880873, 0.6158730158730159, 0.5372983870967742, 0.9449975815753247, 0.9611563326493996, 0.5, 0.9054278472161525, 0.3425219941348973, 0.8108066502463054, 0.8727732686834616, 0.8879620123203285, 0.9167149719829166, 0.8733268372737063, 0.857191152322998, 0.48894404332129965, 0.6871165644171779, 0.9540227225779798, 0.9103278406401928, 0.6477475071225072, 0.6795238095238094, 0.9634814440199558, 0.8081113801452785, 0.8958282803783653, 0.6862348538845332, 0.9505086071987481, 0.9293751045675087, 0.9465120523095083, 0.936077902833151, 0.9313597408026756, 0.9269387755102041, 0.8690422845038293, 0.9510023635179745, 0.8720388104838709, 0.8620370370370369, 0.9434831740257366, 0.8213296538896977, 0.8861326557903183, 0.7942941415785192, 0.5, 0.8928256070640177, 0.9510466988727859, 0.9070022883295195, 0.8039672933190002, 0.6998420221169037, 0.6544901065449011, 0.5669191919191919, 0.9803950992161571, 0.8968357391071906, 0.8203737498343308, 0.9562224978027851, 0.89206110693714, 0.7718025487486564, 0.9873015873015873, 0.5480769230769231, 0.9756834854305614, 0.9730173876166244, 0.9448144944252211, 0.9709639953542393, 0.8390836071558962, 0.9623047904124487, 0.5336832895888014, 0.8846938775510204, 0.958011440862307, 0.7169631744637236, 0.9348378526668399, 0.9720348382164238, 0.7700520833333333, 0.9679431245918644, 0.8058410380540455, null, 0.3425219941348973, 0.6860496132301948, 0.9125066827051591, 0.9353648675965232, 0.6158730158730159, 0.8787263815726813, 0.9068649885583523, 0.9784708000847397, 0.5393145161290323, 0.8831906926985825, 0.9452710495963091, 0.6476584757834758, 0.9249124396738779, 0.48, 0.8013962129997207, 0.8042898193760262, 0.9867647058823529, null, 0.8269594745045646, 0.9623832400356103, 0.8105022831050228, 0.8752327746741154, 0.9384028507920827, 0.6773809523809522, 0.9541077446739712, 0.9773639336770397, 0.9691255388405032, 0.8879073486025719, 0.8795918367346939, 0.7158649974375869, 0.8846751960479393, 0.972689378267219, 0.9265433389074694, 0.683275946876537, 0.9766207951070336, 0.8467672342261608, 0.8529647435897436, 0.5336832895888014, 0.9493593505477308, 0.7656363946686527, 0.888246739486409, 0.7293709150326798, 0.8887878242074927, 0.8224484386310114, 0.8900476998948986, 0.6919208838203849, 0.8937566137566137, 0.9700752756573368, 0.9647529680826018, 0.925232622488771, 0.6998420221169037, null, 0.9521644845152673, 0.9518518518518518, 0.9325857420997454, 0.5, 0.9443725721281444, 0.9192354826546003, 0.5515734265734266, 0.9700434086369079, 0.9076496149378308, 0.8071207491766199, 0.4894837476099426, 0.8698336693548386, 0.8917587802341396, 0.9055219756207433, null, 0.7131026323572908, 0.8004439063761098, 0.9333945615794634, 0.8559900433595633, 0.8704273260550758, 0.7182291666666665, 0.9037944702242846, 0.9364197530864198, 0.4979693140794224, 0.6105703137635452, 0.7902330582354992, 0.9559479446395373, 0.8983995584988962, 0.967030444034173, 0.5655876951331497, 0.5, 0.8607804232804233, 0.9710161014403803, 0.7544270833333333, 0.9740052415455959, 0.9659828307375413, 0.8516479388690341, 0.8696322395948028, 0.9606095488236231, 0.8931960901876016, 0.7487341772151899, 0.7903168022928503, 0.9796931407942238, 0.9709639953542393, 0.9574398280587838, 0.49410377358490565, 0.9667341681299104, 0.9473990941900305, 0.8889039028894444, 0.9575596791227907, 0.7420886075949368, 0.8621031746031745, 0.9520128824476651, 0.9144314362801559, 0.9232804232804231, 0.9195654223227753, 0.9353018402280846, 0.9163325314087142, 0.7312914087107635, 0.8900126080691643, 0.9658810636890549, 0.9687684685569924, 0.7298815359477124, 0.9691651271223718, 0.5480945821854912, 0.8922737306843267, 0.9351851851851852, 0.8004887660576283, 0.9779589932544046, 0.7062742514093271, 0.9726098414261994, 0.9541304622977431, 0.8569711538461539, 0.8613130206777523, 0.8821428571428571, 0.7544270833333333, 0.5336832895888014, 0.9670150987224158, 0.8120907900480139, 0.8508537514622707, 0.9326450975549518, 0.8568866763021252, 0.9713254523607577, 0.5, 0.9502396322378717, null, 0.947046175762654, 0.7142857142857143, 0.9603593272171252, 0.6998420221169037, 0.6834071159206427, 0.9730766742047682, 0.8800038650960689, 0.5340909090909092, 0.8690146169354839, 0.8971650173843273, 0.9224389695971242, 0.9718061558953733, 0.7731314657677555, 0.8002323494807053, 0.9547630654823384, 0.5, 0.7036543078154174, 0.6476584757834758, 0.4970667870036101, 0.9081981671901218, 0.9405600307377278, 0.8725727825756341, 0.9075057208237987, 0.3425219941348973, 0.9672904758361063, 0.9796931407942239, 0.9879785247432307, 0.8859758375887408, 0.9060934695057586, 0.9717722038537917, 0.8740732932785216, 0.8836818981839516, 0.9479270887240527, 0.8929669373549883, 0.539986559139785, 0.48, 0.6776190476190476, 0.9711753869651493, 0.6158730158730159, 0.7333333333333334, 0.8905295806122252, null, 0.8133575464083939, 0.826917724337564, 0.4904397705544933, 0.9246912845394526, 0.8694120237833076, 0.9236430462653289, 0.9815161358661111, null, 0.49410377358490565, 0.9468906189926951, 0.959962386206114, 0.7176069137562369, 0.8204553100818661, 0.8717762147303743, 0.5538977842471293, 0.9332098818909007, 0.8072146962233169, 0.96528487848517, 0.8105022831050228, 0.957532848835727, 0.8501990834648034, 0.5, null, 0.9774353408263236, 0.9090331639475796, 0.9717301687215243, 0.9621240984984206, 0.8371975806451613, 0.8790816326530613, 0.48967936276888246, 0.8932844015488643, 0.9808452792881859, 0.8604828042328042, 0.9270899470899471, 0.8116438356164384, 0.9326539894308238, 0.5620752984389348, 0.946376811594203, 0.9576494743544762, 0.5305944055944056, 0.8203661035611243, 0.8017855782497185, 0.7861456574031425, 0.8698604095377764, 0.9161446249033256, 0.6997432859399685, 0.9448754650712758, 0.9876517273576098, 0.8915562913907283, 0.6289829768659974, 0.9471645520953478, 0.7637597349761711, 0.9662720434085583, 0.9777293577981652, 0.9179697941726811, 0.685, 0.7944889162561577, 0.682243406985032, 0.9547366246643255, 0.49235181644359466, 0.8730374322904034, 0.49410377358490565, null, 0.8665492182338692, 0.9237662642694431, 0.7622395833333333, 0.9252658379511758, 0.9544927641417945, 0.959046803652968, 0.9161481900452488, 0.8801476998820452, 0.9686282853024668, 0.7901533494753834, 0.6932515337423313, 0.5, 0.918391164994426, 0.9643137685704599, 0.972909620382099, 0.48, 0.9499078292693651, 0.5386424731182796, 0.9331141930835736, 0.9830776173285198, 0.8725624538842627, 0.9131674052402875, 0.6476584757834758, 0.47405234657039713, 0.9275275895536144, 0.3425219941348973, 0.9054144003012108, 0.880765153737433, 0.970102832925334, 0.9059038901601831, 0.7284517973856209, 0.7395833333333334, 0.7656363946686527, 0.7609177215189873, 0.9679150763358778, 0.5336832895888014, 0.8469434184465501, 0.6812264305623872, 0.9395828246833433, null, 0.8695111022602807, 0.8403612781117791, 0.9721376961136687, 0.7042243209605389, 0.8561698717948718, 0.967711962833914, 0.7142857142857143, 0.943368544600939, 0.6142857142857142, 0.8825312755722908, 0.9276234567901235, 0.8989905962542799, 0.9666627806853274, 0.9737200813888243, 0.9709509985044428, 0.8903396166925658, 0.9347383901103236, 0.9790162454873647, 0.8941822766570605, 0.8476526952518603, 0.8916794206282838, 0.9504469434832756, 0.9584392997808024, 0.9481080792426121, 0.9793755737589154, 0.9106051817478733, 0.9549531088618054, 0.9104891795379254, 0.8388282516071562, 0.7532622693913016, 0.4933078393881453, 0.8521634615384616, 0.6996445497630333, 0.973195460702687, 0.8271264751725673, 0.971356375459429, 0.8182001614205003, 0.9577804107619858, 0.7174839629365646, 0.7900480139676996, 0.8125, 0.9073226544622426, null, 0.6951186983195519, 0.5469237832874196, 0.6852380952380952, 0.8582341269841269, 0.8685105846774193, 0.8573045517018353, null, 0.9110886603776013, 0.9259673833620944, 0.7756741834243868, 0.9256215366182418, 0.9166430995475113, 0.9682757363569432, 0.7292687908496733, 0.49002304706453176, 0.880627435083215, 0.9678939034045921, 0.9724985816489606, 0.8742192167725816, 0.5303151709401709, 0.8723867748849303, 0.9051324503311258, 0.9348313593797541, 0.7755208333333334, 0.49410377358490565, 0.9195987654320987, 0.971491214708027, 0.9053606126414447, 0.5029332129963899, 0.7142857142857143, 0.9667828106852497, 0.9012206992186458, null, 0.9492942597083686, 0.9729444437189583, 0.9711902863845303, 0.6158730158730159, 0.9178685897435898, 0.48, 0.7863503761707356, 0.5366263440860215, 0.8818231778940662, 0.73671875, 0.9622605565868725, 0.7753164556962024, 0.9197883597883597, 0.9655617365081075, 0.9507246376811594, 0.9469680022586557, 0.9526987767584099, 0.8116438356164384, 0.8880059334604788, 0.9613396157549777, 0.9327406220220047, 0.8927051204091052, 0.5603146853146853, 0.625, 0.8729591836734695, 0.9774147120943082, 0.3425219941348973, 0.7981649046478361, 0.695438904751446, 0.9187215985030741, 0.905210840688241, 0.9669769279924662, 0.9411922926447575, 0.8694120237833076, 0.5, 0.9877450980392156, 0.9391756694774723, 0.6790129529431054, 0.8615702479338843, 0.5, 0.8152316311030003, 0.9273148148148148, 0.8419477844578045, 0.967479674796748, 0.8905118414616829, 0.8501017076173654, 0.967877507705275, 0.6884899163797343, 0.8785714285714286, 0.6579155611413677, 0.7122153209109732, 0.7776898734177216, 0.816796568352585, 0.8276774311257069, 0.7525078049030145, 0.7315104166666666, 0.9708319020415419, 0.5649057900695456, 0.9623983657455071, 0.8732551192222042, 0.8700726712177934, 0.9574472302426255, 0.5, 0.7591973730094153, 0.3425219941348973, 0.8638227513227513, 0.8724168346774194, 0.8177966101694915, 0.9192430394431556, 0.9038901601830664, 0.7897603269711521, 0.5400367309458218, 0.8944812362030905, 0.8612602844101896, null, 0.9611144115180318, 0.9711684535391392, 0.9679425012852335, 0.9722542607276251, 0.9667502419283892, 0.8812902173814579, 0.95184250764526, 0.5, 0.9687765055131468, 0.9381113067292646, 0.9389466182126098, 0.8703989511766073, 0.9490871498203685, 0.6927300680869757, 0.5, 0.9773004606554542, 0.729983660130719, 0.5568181818181819, 0.7010268562401264, 0.9663715400396864, 0.945016456569558, 0.9484702093397746, 0.8907936507936508, 0.810882800608828, 0.9780515147235365, 0.8821398986688244, 0.9388662681567472, 0.8141860150159554, 0.9489571318723568, 0.9874649859943976, 0.9030266559686191, 0.8609775641025641, 0.8935094475598495, 0.967148697729256, 0.9136029411764705, 0.745703125, 0.9572765957446808, 0.49410377358490565, 0.9036797482737524, 0.9646076593474517, 0.47382671480144406, 0.49282982791587, 0.9794675090252707, 0.8882186858316222, 0.9464185329125613, 0.6142857142857142, null, 0.963324682770023, 0.9110315635451506, 0.48, 0.5393145161290323, 0.647480413105413, 0.9081013638514821, 0.8731948982818594, 0.9271378156224945, 0.688316884502534, 0.676190476190476, 0.6706664290805416, 0.9138432237369687, null, 0.861039696014229, 0.9207346108801688, 0.8123973727422003, 0.875756484149856, 0.9359695286176315, 0.9146697618691029, 0.9671801253377281, 0.8921452049497292, 0.8633006820862609, 0.9711218641322783, 0.858531746031746, null, 0.7770569620253164, 0.9027586346002568, 0.6158730158730159, 0.8344884212870185, 0.647636217948718, 0.5, 0.5, 0.8116438356164384, 0.9074284568066328, 0.8561698717948718, 0.7952073070607553, 0.49235181644359466, null, 0.8945623043488051, 0.5585664335664335, 0.9682869079546522, 0.9403853677621283, 0.9376586811558408, 0.8695028046068574, 0.7122153209109732, 0.6936086097078848, 0.9681765389082462, 0.9766498621842019, 0.9667346824788319, 0.9431144672981736, 0.9592592592592593, 0.7570883873279084, 0.9310052910052911, 0.8859157371554066, 0.8927704194260486, 0.8055501477048611, 0.5330348943985307, 0.9851423902894489, 0.868951612903226, null, 0.9299382716049382, 0.9797247706422019, 0.9368823178084044, 0.8980867774459567, 0.9072690217391304, 0.9506296360186304, 0.48, 0.5379704301075269, 0.9005023467292461, 0.8750570956747831, 0.9209291944005811, 0.49751805054151627, 0.6716457722059216, 0.9522683583237217, 0.9676079382778484, 0.8587436332767401, 0.49410377358490565, 0.9127094567637482, 0.9203047775947282, 0.7861979166666668, 0.9130608974358975, 0.7276348039215687, 0.9681902914667613, 0.9683469692970881, 0.679047619047619, 0.7697314890154596, 0.9375667792890352, 0.9717626267242992, 0.790137296139042, 0.8121468926553672, 0.9154799294411221, 0.6290785374651582, 0.9581298167238187, 0.8361398199126394, 0.8669896498568597, 0.9776624548736462, 0.5, 0.9817103311913, 0.9516281759966949, 0.955411794179153, 0.8615314220798688, 0.3425219941348973, 0.8545045768427815, 0.728125, 0.8873829250720461, 0.6168460441910192, 0.9602165066940017, 0.5676401018922852, 0.6868415336157272, 0.972177864242383, 0.8742294009016348, 0.9012814645308924, 0.9688427692960135, 0.7013230647709321, 0.8857142857142857, 0.9157812082330927, 0.9558767499736542, 0.9480560057283816, 0.8206643082161754, 0.901790456724466, 0.8169176343450203, 0.9614298915913397, 0.49887184115523464, 0.49410377358490565, 0.8955272140590416, 0.883673469387755, 0.9691202217090444, 0.6829316281357599, 0.6174603174603175, 0.8941923761658107, 0.8640542328042328, 0.3425219941348973, 0.7669417645007556, 0.8699995926348378, 0.8515624999999999, 0.5373737373737373, 0.9447557391077873, 0.9557673132888562, 0.8986203090507726, null, 0.7298136018114185, 0.7846354166666668, 0.743666513127591, 0.73359375, 0.9716608594657374, 0.9712817047307731, 0.7142857142857143, 0.7650316455696203, 0.866329002422374, 0.5, 0.9386515511781742, 0.9660228394191457, 0.9481586768035131, 0.917283950617284, 0.8958797828975519, 0.9155779682274249, 0.9884920634920635, null, 0.9477508650519031, 0.8561698717948718, 0.9710737236070899, 0.7009281200631912, 0.9507623532979941, 0.970600968334747, 0.8088567323481115, 0.9714858089100736, 0.8735860951008646, 0.7274055595153243, 0.8874607408682182, 0.7865368500351275, 0.49760994263862335, 0.9923285198555957, 0.954756455277835, 0.8622394641758311, 0.9002706883217324, 0.8547856110486591, 0.7804650609102812, 0.95631288669214, 0.8492707427798706, 0.8412936985081274, 0.8036723163841808, 0.6840835389222485, 0.9024713958810069, 0.9158313285217856, 0.9661515703115187, 0.9010542034412052, 0.9331196396084207, 0.8959717967982431, 0.48, 0.647636217948718, 0.5379704301075269, 0.8729840834826604, 0.907897863683508, 0.6952520672179248, null, 0.672142857142857, 0.861282237852696, 0.910632585166214, 0.9217275722629923, 0.9605793083543616, 0.9125424208144797, 0.7266135620915033, 0.9654086390428434, 0.8588741699857172, 0.5, 0.955, 0.9419014084507042, 0.9375740747274459, 0.8730956885392811, 0.625, 0.6898748078190204, 0.9770243530115569, 0.5620629370629371, 0.9425110655641876, 0.7347792998477931, 0.9510466988727858, 0.8990173847316705, 0.9728073268300751, 0.9674141526250294, 0.7853623914495659, 0.9027231121281465, 0.8421164455320053, 0.8641810602357287, 0.7812500000000001, 0.9709963422391857, 0.5, 0.8871688741721854, 0.9428835827044955, 0.8923469387755102, 0.679642857142857, 0.5568181818181819, 0.8883605837443416, 0.9359103663552666, 0.8709535344637744, 0.891682442025556, 0.9350898281762692, 0.943820127755655, 0.966328009715711, 0.970499419279907, null, 0.9925541516245487, 0.8621187369590753, 0.9857434640522877, 0.9574442426938178, 0.9545755397751483, 0.9638244135988391, 0.946787518619186, 0.9734526163406538, 0.5, 0.8575721153846154, 0.9557132275544813, null, 0.9212191358024691, 0.9695094778872911, 0.9418280516431925, 0.9158771681749623, 0.7683315138282387, 0.9519113724256826, 0.8636077089337175, 0.9164827728739433, 0.9653976422978798, 0.6596081324807346, 0.940930756771867, 0.959197247706422, 0.7666139240506329, 0.9511272141706926, 0.7688021393111213, 0.9642929713337594, 0.9229868341462787, 0.8112633181126332, 0.6707075116066822, 0.4952198852772467, 0.8592368016816234, 0.802493842364532, 0.9202954171034895, 0.7868047930031401, 0.8857370509048765, 0.9356933495047504, 0.811896981014628, 0.9677540079496969, 0.9449277347647702, 0.9339725105728567, 0.6353563791874555, 0.8702434910930748, 0.5369623655913978, 0.8784967237903225, 0.8143161561185364, 0.7042243209605388, 0.8813611755607114, 0.7700693072184122, 0.7537760416666666, 0.4811111111111111, 0.7045212056548413, 0.5, 0.8714335421016006, 0.9137513365410318, 0.9677490802472095, 0.8917247826567161, 0.7132505175983437, 0.9752800350371079, 0.6477475071225072, 0.8916931216931216, 0.8151188485732054, 0.9018946703152628, 0.3425219941348973, null, 0.7965092816787732, 0.8619750544684083, 0.49410377358490565, 0.6748107030365094, 0.9183040691192865, 0.7210988562091503, 0.7004344391785151, 0.8745560755418368, 0.6142857142857142, 0.4984205776173285, 0.9529369964883289, 0.8566798941798941, 0.5508608815426997, 0.814039408866995, 0.9197993110678448, 0.7889234801566571, 0.7416666666666667, 0.911906108597285, 0.7329452614379085, 0.9550101170053664, 0.4250313359210738, 0.9133320396570582, 0.9050305810397553, 0.946498435054773, 0.9338083696095677, 0.7568455640744797, 0.7267735558972107, 0.873409329776053, 0.9703723803493873, 0.6618039443155453, 0.494263862332696, 0.5521428571428572, 0.8685923152358027, 0.6899173112830089, 0.4964622641509434, 0.8549135027972408, 0.904279821349143, 0.9493028375733855, 0.49887184115523464, 0.8313888370356541, 0.9385246594756584, 0.48533724340175954, 0.9603514545272513, 0.923085632664437, 0.48, 0.5352822580645161, 0.5, 0.865254207512034, 0.8493589743589743, null, 0.8504489618243642, 0.8722748293327461, 0.576048951048951, 0.5733303635067712, 0.9344316245326604, 0.9007012864088113, null, 0.9484637867417617, 0.9035240274599542, 0.6142857142857142, 0.9855595667870036, 0.9531369551745507, 0.7449367088607595, 0.46618359393489966, 0.8631363417375945, 0.6327229078211183, 0.6384156464801625, 0.9545541689346135, 0.9042781359886508, 0.8490775620593065, 0.8036723163841808, 0.7347792998477931, 0.8499459654178674, 0.5793269230769231, 0.9136761561079926, 0.865410052910053, 0.9434782608695651, 0.806182917611489, 0.9576718909404813, 0.5841544199510132, 0.8769962819549597, 0.924701839498686, 0.5595284448244047, 0.7723033415633036, 0.9776493930905696, null, 0.9125607741107116, 0.8764273356401384, 0.9718725725584351, 0.961770921380648, 0.7055687203791469, 0.9387731273523435, 0.964204304495335, 0.9670525692834647, 0.6996093750000001, 0.8800425670197294, 0.5, 0.9315154554367303, 0.8644472911306462, 0.8944778505956805, 0.9309413580246914, 0.5586547291092746, 0.7490531756998822, 0.7122153209109732, 0.6213231677324151, 0.6594632056451613, 0.5, 0.806136389123941, 0.9616724738675959, 0.871938775510204, 0.8706401766004415, 0.7702037080088341, 0.9138380847812682, 0.6860496132301948, 0.9062362030905077, 0.8631751049641887, 0.8474489795918366, 0.9348827766400679, 0.6051646008803572, null, 0.8808243342081488, 0.9058352402745996, 0.9395478397413258, 0.8440537821396922, 0.5, 0.9008487654320988, 0.8593513931344646, 0.7957532609777955, 0.8673878205128205, 0.9774368231046932, 0.7696318723567859, 0.5396505376344086, 0.9635593370087644, 0.5, 0.7337652829636138, 0.9482099057797235, 0.848931953314248, 0.9486013302034428, 0.5183333333333334, 0.625, 0.9307644937098618, 0.7005718954248367, 0.9075614808874632, 0.8572255291005291, 0.956431568586332, 0.41671720847485033, 0.9067920437405732, 0.690700216505772, 0.5681818181818181, 0.8628688033548001, 0.7142857142857143, 0.851648537260996, 0.7385416666666667, 0.49713193116634796, 0.763625720461095, 0.8032014945915361, 0.8672949833335479, 0.956794425087108, 0.891039755351682, 0.7199302134646963, 0.560520313613685, 0.828959495512976, 0.7046800947867299, 0.6142857142857142, null, 0.8883423628511974, 0.5074149265274555, 0.5783253205128206, 0.48, 0.7669127048703941, 0.8751582123275393, 0.4964622641509434, 0.6594133076387774, 0.7072339122689824, 0.9841619981325863, 0.9155879711916678, 0.9210127349909035, 0.855151137363663, 0.6134462908656457, 0.7199783139062078, 0.5925360104757748, 0.6822546787313464, 0.7736507936507935, 0.955877616747182, 0.9516830133880035, 0.4986462093862816, 0.7011067388096408, 0.9026981234830198, 0.9222308033940142, 0.9487511446654846, 0.8668501883512025, 0.5664335664335666, 0.7632326012444528, 0.48240469208211145, 0.7631154156577885, 0.8484944807258856, 0.6427041330645161, 0.882318529229498, 0.9604531339010995, 0.8305379746835443, 0.8162100456621004, 0.7700969941911715, 0.646875, 0.8564901488811626, 0.9157347408026755, 0.4184193502579719, 0.7001253902451507, 0.9500057455801073, 0.9230975106773798, 0.9614026840086886, 0.8528815806749015, 0.8471939329748519, null, 0.8829016142539491, 0.9491289198606271, 0.7544105854049721, 0.6036011824778286, 0.4976099426386233, 0.5510389477455346, 0.6203325774754346, 0.7982760568931908, 0.8543154761904762, 0.5791711182336183, 0.879955133280549, 0.941387910798122, 0.7332860164773993, null, 0.7346804304853943, 0.9507346208835762, 0.9139869012983353, 0.6636697602474866, 0.7356593617839292, 0.6549918682996447, 0.5, 0.9115951636572279, 0.8459183673469387, 0.8178781968096643, 0.9184211706409265, 0.5874125874125874, 0.7634977722873977, 0.9136321195144723, 0.10065337754034971, 0.9181950924488932, 0.8171871689852411, 0.8226196295692791, 0.5269656596778426, 0.8514062992241361, 0.7045813586097947, 0.8871175523349436, 0.8600840978593274, 0.8487795222806338, 0.8467991169977925, 0.7375810518731989, 0.9618762369239469, 0.6368795481698708, 0.9593285190070125, 0.7040826364977865, 0.43243030025718293, 0.8785970596432553, 0.8444395785994628, 0.7457278481012658, 0.4777777777777778, 0.42705300824842307, 0.98014440433213, 0.5352822580645161, 0.47580645161290325, 0.8627804487179487, 0.8973765432098766, 0.8773584905660378, 0.5, 0.8293888024620047, 0.8509211999445284, 0.48533724340175954, 0.7359208523592086, 0.640625, 0.8412215460897002, 0.5, 0.75425723584796, 0.6142857142857142, 0.9502473363774734, 0.5262762419680572, 0.936234709761194, 0.9054076768437401, 0.49887184115523464, 0.5916431887623507, 0.6094779194111843, 0.5988571275627641, 0.9012770794330034, 0.9206831269227735, 0.8566820133211627, 0.5, 0.9103432494279176, 0.8305306559761135, 0.7699757869249395, 0.7111979166666667, 0.7243896959712446, 0.7408278761419678, 0.6019054178145087, 0.6865751960085531, 0.6763649425287356, null, 0.5949353676626403, 0.6376190476190475, 0.9170437405731524, null, 0.7122153209109732, 0.8467297951992954, 0.7919334366429802, 0.8587316611452911, 0.8622784496543027, 0.6428380901573753, 0.8570308854340832, 0.7018995098039216, 0.9462158962397511, 0.8992019255982195, 0.61422791410264, 0.5, 0.5830919407960001, 0.8716393811955543, 0.753056933322588, 0.8457671957671958, 0.8667807329117214, 0.7577734511217018, 0.6142857142857142, 0.9212676996496464, 0.6419383856738534, 0.7325010463955697, 0.5572314049586776, 0.5494416426512969, 0.9286531279178337, 0.7629702341085765, 0.4959677419354839, 0.5783253205128206, 0.5648548635140331, 0.77467409542658, 0.6966713155850576, 0.7360108303249098, 0.8734303425649087, 0.8354856512141279, 0.45933544303797463, null, 0.8646224256292907, 0.9093906217723611, 0.5242426942266573, 0.7273899872197416, 0.6488968021226086, 0.7047788309636651, 0.7952085004009622, null, 0.7822953092662638, 0.801062484402237, 0.5134227289759649, 0.8416784509411043, 0.9237363711521465, 0.506029782889287, 0.6840490797546012, 0.5542857142857143, 0.6785714285714286, 0.7142857142857143, 0.7173659956342469, 0.7208333333333333, 0.7919975289893307, 0.6867230792271979, 0.8467297951992954, null, 0.7007761437908496, 0.494263862332696, 0.4317483422286916, 0.7640670361572974, 0.9365853658536585, 0.5927004037213629, 0.837454128440367, 0.0800360465524742, 0.9067920437405732, 0.6029080910849484, 0.8572432451424756, 0.7415895061728395, 0.7385844748858448, 0.640625, 0.9438955091168887, 0.5, 0.7837308289980408, 0.5935271561082711, 0.4972924187725632, 0.7029954533297672, 0.9004637194469174, 0.715594082508867, 0.4868035190615836, 0.8450354645912186, 0.9428831981752571, 0.701574028529267, 0.9051613015607581, 0.5, 0.4777777777777778, 0.8473557692307692, 0.5406586021505376, 0.9040003912363067, 0.8342224182337791, 0.7487893462469734, 0.625, 0.9178743961352657, 0.8471260387811635, 0.8229118504919117, 0.6950728457427338, 0.6926352705410821, 0.5430104408352666, 0.4825174825174825, 0.6049736134621129, 0.5861980347694634, 0.9372463536586733, 0.9035017890913761, 0.5954501253902451, 0.8321428571428571, 0.9796931407942239, 0.9038477169026559, 0.6865156945802107, 0.7271710661260038, 0.7566758369725154, 0.9237521274882361, 0.6739541330645161, 0.8285714285714287, 0.7824482347303923, 0.5661264929194776, 0.7609567901234568, 0.8617624457756662, 0.8918530518394648, 0.8407451923076922, 0.8948499392857022, 0.5379704301075269, 0.7845261336116529, 0.8545529609785087, 0.48, 0.5232594936708861, 0.5, 0.8449984188890061, 0.681926518040481, 0.875868511706662, 0.6142857142857142, 0.7790556900726392, 0.7072560825096322, 0.7989860285875466, 0.640625, 0.8587653954275773, 0.5512622826908541, 0.5, 0.502200726751625, 0.9142649869960079, 0.9206119162640902, 0.620809273894078, 0.7965621692228002, 0.7336377473363774, 0.9041148762100091, 0.710893002441009, 0.578042537994524, 0.8397008287851223, 0.49774368231046934, 0.7392462333957388, 0.48240469208211145, 0.899457416982109, 0.5112681333975697, 0.4475478735248274, 0.5, 0.8380880640722578, 0.5738832853025938, 0.6364567961960977, 0.7644355203828894, 0.6533803127064524, 0.5178145087235997, 0.494263862332696, 0.5117863496684458, 0.9367976039016117, 0.6181576797385621, 0.8339653025059461, 0.825450817446092, 0.7744215712149204, 0.8794425087108013, 0.6972285089066365, 0.6425067720916612, 0.5668044077134986, 0.49817162838309137, null, 0.8196593915343916, 0.7705326116011761, 0.7054699842022116, 0.6606574923547401, 0.5, 0.08308676302499207, 0.5431216163959782, 0.4644542378917379, 0.6836999505684626, 0.771493544600939, 0.6549932718185314, 0.7946431776403664, 0.908913084464555, 0.5495260156806843, 0.6083916083916083, 0.6161904761904761, 0.6118123973727423, 0.6700453454254468, 0.6317395536917777, 0.791887417218543, 0.941152574862462, 0.8733704771329003, 0.6971460910340366, 0.7354166666666666, 0.7122153209109732, null, 0.902974828375286, 0.5366121481823961, 0.8661064425770308, 0.7931067489478901, 0.7108066210486806, null, 0.5688054882970137, 0.519958430007734, 0.36791638833222007, 0.7440476190476191, 0.735632183908046, 0.5791711182336183, 0.7423896499238964, 0.46776098778639574, 0.6428067153873606, 0.7926800887397474, 0.8469551282051282, 0.7110893408103056, 0.4980879541108987, 0.8243707093821511, 0.5, null, 0.8567201834862387, 0.676180817154448, 0.9089151337792643, 0.4822222222222222, 0.5413306451612904, 0.8708759354906714, 0.850143140277472, 0.5870106449370477, 0.6657108562283275, 0.5464285714285714, 0.73046875, 0.6766432209556148, 0.7519473687597328, 0.8816085068144378, 0.8989206259426848, 0.5123322011968301, 0.6204044117647058, 0.8211665347057271, 0.6142857142857142, 0.08796412638450915, 0.8780320813771517, 0.5, 0.8822122395323226, 0.8505870176257496, 0.5809488253020081, 0.6514951179820992, 0.7328761711541795, 0.5030241935483871, 0.5066804407713499, 0.7045253863134658, 0.8214285714285715, 0.5516403091253391, 0.8118334456452593, 0.815911730545877, 0.7758961295557983, 0.7122153209109732, 0.745216049382716, 0.8610898252177734, 0.8153044615049411, 0.8355300453514739, 0.653125, 0.5, 0.6164901687190392, 0.4972924187725632, 0.5, 0.6358009509755698, 0.5390321617046699, 0.5310943959126609, 0.7064573459715641, 0.6491508368053103, 0.7817880614849819, 0.8242724867724869, 0.6440301772653232, 0.9219015055131468, 0.7937078815667915, null, 0.542768371757925, 0.6248218104062723, 0.660585982876565, 0.8309850420459901, 0.5867088607594937, 0.5051895306859205, 0.6879809956620533, 0.7279224979882926, 0.8972857402190503, 0.4983673250896633, 0.6213939923018929, 0.9185643809874414, 0.7847126105344098, null, 0.5, 0.815558005671262, 0.8591898451776365, 0.6294179894179893, 0.6048951048951048, 0.5816139783081931, 0.908695652173913, 0.7375286644736361, 0.8747389999584704, 0.5405892111967902, 0.888141923436041, 0.941187485024334, 0.6779412841350025, 0.7477836495422912, 0.5, 0.8173469387755102, 0.8803493508501812, null, 0.7751978721734548, 0.6142857142857142, 0.5692490572224955, 0.07778921683865789, 0.42000242600679283, 0.844747852895838, 0.8839125547479265, 0.780256130351007, 0.5783253205128206, null, 0.4868035190615836, 0.7114583333333334, 0.5, 0.7587381483233115, 0.8192294973544972, 0.7269889110250121, 0.5806962025316456, 0.7412891749497582, 0.6975620144965202, 0.7826625084307713, 0.8916396669453734, 0.5296171523857087, 0.918722331770636, 0.8681489262371614, null, 0.7233227604767397, 0.5877664917959594, 0.7951724807570089, 0.7182508200218672, 0.6641982622432859, 0.6643836929946224, 0.4979693140794224, 0.8843564461407972, 0.9004996229260935, 0.5668638305132254, 0.5872658501440923, 0.6126318602993586, 0.5, 0.7900873381233059, 0.7471211781377756, 0.6549479166666666, 0.5497986861623225, 0.5103250188964474, 0.6948529411764707, 0.5108179109663968, 0.751843054455881, 0.6969682270191608, 0.563686516551972, 0.7977446483180428, 0.545, 0.8730927230046948, 0.7224223265003704, 0.6461723126166978, 0.541637624879684, 0.6489128047441247, 0.6485893737187985, 0.9319553340347103, 0.8471592705807948, 0.5926573426573426, 0.8592664044829714, 0.8473429951690822, 0.5369623655913979, 0.48, 0.7951267443366767, 0.741481952972892, 0.5051895306859205, 0.7514518002322881, 0.8378032036613271, 0.7458400160384924, 0.8348097642757913, 0.5301064537591484, 0.49581106657759966, 0.668598233995585, 0.594272041763341, 0.497131931166348, 0.3800287356321839, 0.5792929292929293, 0.8263221153846153, 0.49798387096774194, 0.45725238477311225, 0.7351598173515982, 0.6256974311286762, 0.5, 0.6438075067107324, 0.5924132364810331, 0.5871181726982644, 0.5645709544456934, 0.7142857142857143, 0.8894870031201626, 0.8670737450939888, 0.7165507837591139, 0.7277371053503526, 0.8981686503696898, 0.6218503233998651, 0.7634582568659648, 0.7615740740740741, 0.6577129709213101, 0.5679981634527089, 0.6546412376633769, 0.7197916666666666, 0.7537746806039489, 0.4964622641509434, 0.480952380952381, 0.8501071661541328, 0.6723381378588864, 0.6844045597865631, 0.6520833333333332, 0.5117936062598937, 0.4964717741935484, 0.8278406337686921, 0.8107585819191174, 0.4868035190615836, 0.6175466011831677, 0.5, 0.5696999508116085, 0.7208058881570176, 0.8942469671050055, 0.7142857142857143, 0.809645430403721, 0.8100526059085841, 0.6415565495644447, 0.4984205776173285, 0.4822222222222222, 0.5096089871538972, 0.8476106584452764, 0.6432098765432098, 0.538978494623656, 0.7923841059602649, 0.631504003531313, 0.7469387755102042, 0.5, 0.8133520274890973, 0.6938183315860417, 0.5246756949394156, 0.5943037974683545, 0.6014714107473218, 0.345014153140564, 0.26173285198555957, 0.6239066308613923, 0.8960419005869579, 0.7275418514018838, 0.5376050532626734, 0.6184079996825523, 0.6150793650793651, 0.9009369848799703, 0.6615903066207326, 0.556286023054755, 0.5797013720742534, 0.6503739316239316, 0.7963157894736844, 0.7533413525795243, 0.8590039905592431, null, 0.5587842351369405, 0.6374564965197215, 0.4933078393881453, 0.5480295566502464, 0.6714974213454045, 0.8357371794871795, 0.5303557100829099, 0.7393455098934552, 0.6439587891200794, 0.8489219948196807, null, 0.764074427480916, 0.8963753770739066, 0.534517973856209, 0.7137679246942905, 0.07765546089057142, 0.5968584016015992, 0.5742660550458715, 0.9017507824726134, 0.7132726359241758, 0.5446825671241575, 0.513101994755883, 0.6642872831100374, 0.9639568446926239, 0.5, 0.597027972027972, 0.8539319353072377, 0.8057971014492753, 0.49507667059849225, 0.4728495842781557, 0.4323913001598952, 0.599490311588553, 0.5726187352400944, 0.7517181525639691, 0.8633870214752568, null, 0.6217704728950404, 0.7804398148148148, 0.7430795847750866, 0.7063586097946288, 0.5968235451353455, 0.4599607795568495, 0.7215195438743156], "y0": " ", "yaxis": "y"}],                        {"boxmode": "group", "height": 500, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Distribution of test_roc_auc by learning rate value"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": " "}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "test_roc_auc"}}},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('6909a156-60bc-4403-81b6-b3dd83cbf1a7');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

<a id='reference17'></a>
#### Frequency of best hyper-parameter values


```python
plt.figure(figsize=(8,5))

best_param_freq = sns.countplot(best_learning_values['param_value'], palette='Greens_r')

for p in best_param_freq.patches:
    best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

plt.ylabel('frequency')
plt.xlabel('learning_rate')
plt.title('Count of best learning rate value', loc='left')
```




    Text(0.0, 1.0, 'Count of best learning rate value')




![png](output_218_1.png)



```python
plt.figure(figsize=(12,5))

best_param_freq = sns.countplot(best_learning_values[best_learning_values.param_value.isin([0.01, 0.05, 0.1, 0.7])]['param_value'],
                                palette='Greens',
                                hue=best_learning_values[best_learning_values.param_value.isin([0.01, 0.05, 0.1, 0.7])]['quartile_n_orders'])

for p in best_param_freq.patches:
    best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

plt.ylabel('frequency')
plt.xlabel('learning_rate')
# plt.legend(loc='upper center', title='quartile_n_orders')
plt.legend(loc='upper left', title='quartile_n_orders', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
plt.title('Count of best learning rate value', loc='left')
```




    Text(0.0, 1.0, 'Count of best learning rate value')




![png](output_219_1.png)



```python
# plt.figure(figsize=(12,5))

# best_param_freq = sns.countplot(best_learning_values[best_learning_values.param_value.isin([0.01, 0.05, 0.1, 0.7])]['quartile_n_orders'],
#                                 palette='Greens',
#                                 hue=best_learning_values[best_learning_values.param_value.isin([0.01, 0.05, 0.1, 0.7])]['param_value'])

# for p in best_param_freq.patches:
#     best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                              ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

# plt.ylabel('frequency')
# plt.xlabel('quartile_n_orders')
# plt.legend(loc='upper left', title='learning_rate', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
# plt.title('Count of best learning rate value', loc='left')
```


```python
plt.figure(figsize=(12,5))

best_param_freq = sns.countplot(best_learning_values[best_learning_values.param_value.isin([0.01, 0.05, 0.1, 0.7])]['param_value'],
                                palette='Greens',
                                hue=best_learning_values[best_learning_values.param_value.isin([0.01, 0.05, 0.1, 0.7])]['quartile_n_vars'])

for p in best_param_freq.patches:
    best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

plt.ylabel('frequency')
plt.xlabel('learning_rate')
plt.legend(loc='upper left', title='quartile_n_vars', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
plt.title('Count of best learning rate value', loc='left')
```




    Text(0.0, 1.0, 'Count of best learning rate value')




![png](output_221_1.png)



```python
# plt.figure(figsize=(12,5))

# best_param_freq = sns.countplot(best_learning_values[best_learning_values.param_value.isin([0.01, 0.05, 0.1, 0.7])]['quartile_n_vars'],
#                                 palette='Greens',
#                                 hue=best_learning_values[best_learning_values.param_value.isin([0.01, 0.05, 0.1, 0.7])]['param_value'])

# for p in best_param_freq.patches:
#     best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                              ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

# plt.ylabel('frequency')
# plt.xlabel('quartile_n_vars')
# plt.legend(loc='upper left', title='learning_rate', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
# plt.title('Count of best learning rate value', loc='left')
```

[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

#### Performance metric against dataset information

When exploring learning rate, the plot of performance metrics against numbers of observations and features produce the same patters as those obtained when exploring [subsample](#performance_data_info)<a href='#performance_data_info'></a>.

#### Distribution of performance metric by best hyper-parameter value


```python
px.strip(data_frame=best_learning_values.sort_values('param_value'),
         x=best_learning_values.sort_values('param_value')['param_value'].apply(lambda x: 'v = ' + str(x)),
         y=best_learning_values.sort_values('param_value')['test_roc_auc'],
         hover_data=['store_id', 'param_value'],
         color_discrete_sequence=['#0b6fab'],
         width=900, height=500, title='Distribution of test_roc_auc by best learning rate value',
         labels={'y': 'test_roc_auc', 'x': ''})
```


<div>                            <div id="79ca6515-1147-47bb-8159-a21180e94bff" class="plotly-graph-div" style="height:500px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("79ca6515-1147-47bb-8159-a21180e94bff")) {                    Plotly.newPlot(                        "79ca6515-1147-47bb-8159-a21180e94bff",                        [{"alignmentgroup": "True", "boxpoints": "all", "customdata": [[9761.0, 0.01], [12658.0, 0.01], [6714.0, 0.01], [5085.0, 0.01], [4601.0, 0.01], [10650.0, 0.01], [7755.0, 0.01], [11729.0, 0.01], [8783.0, 0.01], [7939.0, 0.01], [8832.0, 0.01], [9281.0, 0.01], [1875.0, 0.01], [1961.0, 0.02], [1979.0, 0.02], [2212.0, 0.02], [8421.0, 0.02], [2699.0, 0.02], [6047.0, 0.02], [6966.0, 0.03], [4838.0, 0.03], [7185.0, 0.03], [1559.0, 0.03], [3481.0, 0.04], [6170.0, 0.04], [6083.0, 0.04], [1098.0, 0.04], [6970.0, 0.04], [8358.0, 0.04], [2866.0, 0.04], [6256.0, 0.04], [6106.0, 0.04], [5663.0, 0.05], [9098.0, 0.05], [5593.0, 0.05], [1603.0, 0.05], [3781.0, 0.05], [5847.0, 0.06], [6105.0, 0.06], [7161.0, 0.06], [4636.0, 0.06], [1424.0, 0.06], [8181.0, 0.06], [8282.0, 0.06], [6044.0, 0.06], [7333.0, 0.07], [9177.0, 0.07], [720.0, 0.07], [11723.0, 0.07], [4268.0, 0.08], [3437.0, 0.08], [6004.0, 0.08], [3962.0, 0.08], [4408.0, 0.08], [1181.0, 0.08], [2868.0, 0.08], [9491.0, 0.08], [9541.0, 0.08], [5428.0, 0.08], [6929.0, 0.08], [3859.0, 0.09], [2056.0, 0.09], [8894.0, 0.09], [5848.0, 0.09], [3988.0, 0.09], [4030.0, 0.09], [5342.0, 0.09], [5215.0, 0.09], [7630.0, 0.09], [2352.0, 0.1], [1210.0, 0.1], [8436.0, 0.1], [9409.0, 0.1], [5939.0, 0.1], [10311.0, 0.1], [10349.0, 0.1], [4974.0, 0.1], [7849.0, 0.1], [7845.0, 0.2], [8790.0, 0.2], [5394.0, 0.2], [1739.0, 0.3], [1241.0, 0.3], [4736.0, 0.3], [3146.0, 0.3], [10060.0, 0.4], [11223.0, 0.4], [12980.0, 0.4], [7988.0, 0.4], [7292.0, 0.4], [2782.0, 0.4], [5168.0, 0.6], [10268.0, 0.7], [7790.0, 0.7], [6971.0, 0.7], [8446.0, 0.7], [6078.0, 0.7]], "fillcolor": "rgba(255,255,255,0)", "hoveron": "points", "hovertemplate": "=%{x}<br>test_roc_auc=%{y}<br>store_id=%{customdata[0]}<br>param_value=%{customdata[1]}<extra></extra>", "legendgroup": "", "line": {"color": "rgba(255,255,255,0)"}, "marker": {"color": "#0b6fab"}, "name": "", "offsetgroup": "", "orientation": "v", "pointpos": 0, "showlegend": false, "type": "box", "x": ["v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.01", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.02", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.03", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.04", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.05", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.06", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.07", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.08", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.09", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.1", "v = 0.2", "v = 0.2", "v = 0.2", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.3", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.4", "v = 0.6", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7", "v = 0.7"], "x0": " ", "xaxis": "x", "y": [0.7864237348538845, 0.8031249999999999, 0.7177038817663819, 0.7402380952380953, 0.9749213669776647, 0.8751003693592418, 0.8654431216931217, 0.8034333953272115, 0.9040342792465034, 0.9324832775919732, 0.9439868204283361, 0.8802118548568852, 0.9534917840375587, 0.9213565233785822, 0.8213296538896977, 0.9344650576368876, 0.5336832895888014, 0.9755212457112695, 0.9293751045675087, 0.8224484386310114, 0.9740052415455959, 0.7656363946686527, 0.9364197530864198, 0.8004887660576283, 0.9779589932544046, 0.8120907900480139, 0.9672904758361063, 0.9060934695057586, 0.9144314362801559, 0.947046175762654, 0.9405600307377278, 0.7142857142857143, 0.9544927641417945, 0.9090331639475796, 0.9395828246833433, 0.9737200813888243, 0.970102832925334, 0.9729444437189583, 0.9724985816489606, 0.8182001614205003, 0.625, 0.9577804107619858, 0.5029332129963899, 0.9187215985030741, 0.971491214708027, 0.9572765957446808, 0.9388662681567472, 0.8419477844578045, 0.9192430394431556, 0.8742294009016348, 0.9592592592592593, 0.8980867774459567, 0.9682869079546522, 0.7861979166666668, 0.9817103311913, 0.9797247706422019, 0.9506296360186304, 0.9522683583237217, 0.9154799294411221, 0.9310052910052911, 0.9674141526250294, 0.9614298915913397, 0.8959717967982431, 0.9716608594657374, 0.9884920634920635, 0.8941923761658107, 0.9386515511781742, 0.8958797828975519, 0.6174603174603175, 0.9164827728739433, 0.5, 0.7045212056548413, 0.9677540079496969, 0.9925541516245487, 0.8784967237903225, 0.8143161561185364, 0.8923469387755102, 0.8745560755418368, 0.7329452614379085, 0.814039408866995, 0.8722748293327461, 0.8162100456621004, 0.8673878205128205, 0.9062362030905077, 0.8305379746835443, 0.8773584905660378, 0.7346804304853943, 0.9103432494279176, 0.6019054178145087, 0.5, 0.7040826364977865, 0.6083916083916083, 0.5413306451612904, 0.7064573459715641, 0.5, 0.4980879541108987, 0.4822222222222222], "y0": " ", "yaxis": "y"}],                        {"boxmode": "group", "height": 500, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Distribution of test_roc_auc by best learning rate value"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": ""}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "test_roc_auc"}}},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('79ca6515-1147-47bb-8159-a21180e94bff');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


<a id='reference19'></a>
#### Correlation between performance metric of best hyper-parameter value and dataset information


```python
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(best_learning_values[['test_roc_auc', 'n_orders', 'n_vars', 'avg_y']].corr(),
                            dtype=np.bool))

sns.heatmap(best_learning_values[['test_roc_auc', 'n_orders', 'n_vars', 'avg_y']].corr(),
            mask = mask, annot = True, cmap = 'viridis')
plt.title('Correlation between performance metric and dataset information')
plt.tight_layout()
```


![png](output_229_0.png)



```python
# Generate masks for the upper triangle:
mask001 = np.triu(np.ones_like(metrics_learning[metrics_learning.param_value==0.01][['test_roc_auc', 
                                                                                 'n_orders', 'n_vars',
                                                                                 'avg_y']].corr(), dtype=np.bool))
mask07 = np.triu(np.ones_like(metrics_learning[metrics_learning.param_value==0.7][['test_roc_auc', 
                                                                                 'n_orders', 'n_vars',
                                                                                 'avg_y']].corr(), dtype=np.bool))
```


```python
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))

corr_matrices = sns.heatmap(metrics_learning[metrics_learning.param_value==0.01][['test_roc_auc', 'n_orders',
                                                                 'n_vars', 'avg_y']].corr(),
                            mask = mask001, annot = True, cmap = 'viridis', ax=axs[0])
sns.heatmap(metrics_learning[metrics_learning.param_value==0.1][['test_roc_auc', 'n_orders',
                                                                 'n_vars', 'avg_y']].corr(),
            mask = mask07, annot = True, cmap = 'viridis', ax=axs[1])
sns.heatmap(metrics_learning[metrics_learning.param_value==0.7][['test_roc_auc', 'n_orders',
                                                                 'n_vars', 'avg_y']].corr(),
            mask = mask07, annot = True, cmap = 'viridis', ax=axs[2])

axs[0].set_title('learning_rate = 0.01', loc='left')
axs[1].set_title('learning_rate = 0.1', loc='left')
axs[2].set_title('learning_rate = 0.7', loc='left')

plt.tight_layout()
```


![png](output_231_0.png)


[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

<a id='n_estimators'></a>

## Number of estimators

In order to account for the trade-off between learning rate and number of estimators, two different sets of tests were developed: one considering $learning\_rate = 0.05$ and other with a higher value of $learning\_rate = 0.1$, so that large numbers of estimators are less prone to suffer from overfitting. Given that results have not vary substantially, those for $learning\_rate = 0.1$ were choose, since this value was also used for other tests and it is the sklearn default option.

Main findings are listed below:
* Very similar results across different values for a given set of remaining hyper-parameters.
* Moderate values of $n\_estimators$ were prone to perform better.
* Large values of $n\_estimators$ rarely were among the best hyper-parameter value.

<a id='proc_data_n_estimators'></a>

### Processing data

#### Performance metrics


```python
# Assessing missing hyper-parameter values:
for s in tun_n_estimators['005'].keys():
    for m in ['test_roc_auc', 'test_prec_avg', 'test_pr_auc', 'test_deviance', 'test_brier_score']:
        if len(tun_n_estimators['005'][s][m].keys()) != 9:
            print('Missing hyper-parameter value for store ' + str(s) + ' and metric ' + m + '!')
```


```python
# Collecting reference data:
param_value = []
stores = []

# Loop over datasets:
for s in tun_n_estimators['005'].keys():
    # Loop over hyper-parameter values:
    for v in tun_n_estimators['005'][s]['test_roc_auc'].keys():
        param_value.append(float(v))
        stores.append(int(s))

metrics_n_estimators = pd.DataFrame(data=param_value, columns=['param_value'], index=stores)

# Collecting performance metrics:
for m in ['test_roc_auc', 'test_prec_avg', 'test_pr_auc', 'test_deviance', 'test_brier_score']:
    stores = []
    param_value = []
    ref = []
    
    # Loop over datasets:
    for s in tun_n_estimators['005'].keys():
        # Loop over hyper-parameter values:
        for v in tun_n_estimators['005'][s][m].keys():
            stores.append(int(s))
            ref.append(float(tun_n_estimators['005'][s][m][v]))

    metrics_n_estimators = pd.concat([metrics_n_estimators, pd.DataFrame(data={m: ref},
                                                                   index=stores)], axis=1)

metrics_n_estimators.index.name = 'store_id'
metrics_n_estimators.reset_index(inplace=True, drop=False)
metrics_n_estimators.param_value = metrics_n_estimators.param_value.apply(lambda x: x if np.isnan(x) else int(x))
print('\033[1mShape of metrics_n_estimators:\033[0m ' + str(metrics_n_estimators.shape) + '.')
metrics_n_estimators.head()
```

    [1mShape of metrics_n_estimators:[0m (900, 7).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>param_value</th>
      <th>test_roc_auc</th>
      <th>test_prec_avg</th>
      <th>test_pr_auc</th>
      <th>test_deviance</th>
      <th>test_brier_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11729</td>
      <td>100</td>
      <td>0.801450</td>
      <td>0.165249</td>
      <td>0.157897</td>
      <td>885.405667</td>
      <td>0.041386</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11729</td>
      <td>250</td>
      <td>0.771918</td>
      <td>0.140431</td>
      <td>0.131739</td>
      <td>885.743592</td>
      <td>0.044329</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11729</td>
      <td>500</td>
      <td>0.772637</td>
      <td>0.180357</td>
      <td>0.173761</td>
      <td>886.097185</td>
      <td>0.041737</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11729</td>
      <td>750</td>
      <td>0.749840</td>
      <td>0.195791</td>
      <td>0.191797</td>
      <td>886.435896</td>
      <td>0.042187</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11729</td>
      <td>1000</td>
      <td>0.739582</td>
      <td>0.178299</td>
      <td>0.173366</td>
      <td>886.065753</td>
      <td>0.045934</td>
    </tr>
  </tbody>
</table>
</div>



<a id='stats_n_estimators'></a>

### Statistics by hyper-parameter value

#### Basic statistics for each performance metric


```python
# Test ROC-AUC:
metrics_n_estimators.groupby('param_value').describe()[['test_roc_auc']].sort_values(('test_roc_auc','mean'),
                                                                                  ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_roc_auc</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>97.0</td>
      <td>0.829800</td>
      <td>0.154018</td>
      <td>0.342522</td>
      <td>0.759968</td>
      <td>0.885251</td>
      <td>0.947343</td>
      <td>0.987582</td>
    </tr>
    <tr>
      <th>1500</th>
      <td>97.0</td>
      <td>0.827766</td>
      <td>0.155183</td>
      <td>0.342522</td>
      <td>0.734115</td>
      <td>0.890435</td>
      <td>0.945894</td>
      <td>0.987745</td>
    </tr>
    <tr>
      <th>750</th>
      <td>97.0</td>
      <td>0.827545</td>
      <td>0.157232</td>
      <td>0.342522</td>
      <td>0.749840</td>
      <td>0.890548</td>
      <td>0.948953</td>
      <td>0.987558</td>
    </tr>
    <tr>
      <th>1250</th>
      <td>97.0</td>
      <td>0.827459</td>
      <td>0.155144</td>
      <td>0.342522</td>
      <td>0.737330</td>
      <td>0.889709</td>
      <td>0.947633</td>
      <td>0.987768</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>97.0</td>
      <td>0.827296</td>
      <td>0.154771</td>
      <td>0.342522</td>
      <td>0.734635</td>
      <td>0.887308</td>
      <td>0.943932</td>
      <td>0.987488</td>
    </tr>
    <tr>
      <th>250</th>
      <td>97.0</td>
      <td>0.827100</td>
      <td>0.155285</td>
      <td>0.342522</td>
      <td>0.755859</td>
      <td>0.887524</td>
      <td>0.944605</td>
      <td>0.988042</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>97.0</td>
      <td>0.826607</td>
      <td>0.156512</td>
      <td>0.342522</td>
      <td>0.739582</td>
      <td>0.890402</td>
      <td>0.946538</td>
      <td>0.994359</td>
    </tr>
    <tr>
      <th>1750</th>
      <td>97.0</td>
      <td>0.824530</td>
      <td>0.157618</td>
      <td>0.342522</td>
      <td>0.732552</td>
      <td>0.888081</td>
      <td>0.945933</td>
      <td>0.987605</td>
    </tr>
    <tr>
      <th>100</th>
      <td>97.0</td>
      <td>0.820743</td>
      <td>0.150249</td>
      <td>0.342522</td>
      <td>0.735476</td>
      <td>0.869646</td>
      <td>0.936787</td>
      <td>0.978565</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test average precision score:
metrics_n_estimators.groupby('param_value').describe()[['test_prec_avg']].sort_values(('test_prec_avg','mean'),
                                                                                   ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_prec_avg</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1500</th>
      <td>97.0</td>
      <td>0.418514</td>
      <td>0.268164</td>
      <td>0.000960</td>
      <td>0.217415</td>
      <td>0.436735</td>
      <td>0.632634</td>
      <td>0.951868</td>
    </tr>
    <tr>
      <th>500</th>
      <td>97.0</td>
      <td>0.416903</td>
      <td>0.267921</td>
      <td>0.000956</td>
      <td>0.194097</td>
      <td>0.404169</td>
      <td>0.620772</td>
      <td>0.952257</td>
    </tr>
    <tr>
      <th>1750</th>
      <td>97.0</td>
      <td>0.416144</td>
      <td>0.271367</td>
      <td>0.000960</td>
      <td>0.183693</td>
      <td>0.436735</td>
      <td>0.633297</td>
      <td>0.951645</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>97.0</td>
      <td>0.415830</td>
      <td>0.271904</td>
      <td>0.000960</td>
      <td>0.179917</td>
      <td>0.447790</td>
      <td>0.608965</td>
      <td>0.952926</td>
    </tr>
    <tr>
      <th>750</th>
      <td>97.0</td>
      <td>0.415493</td>
      <td>0.271327</td>
      <td>0.000956</td>
      <td>0.195791</td>
      <td>0.421991</td>
      <td>0.607136</td>
      <td>0.954189</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>97.0</td>
      <td>0.415000</td>
      <td>0.267964</td>
      <td>0.000956</td>
      <td>0.205671</td>
      <td>0.436735</td>
      <td>0.621236</td>
      <td>0.950484</td>
    </tr>
    <tr>
      <th>1250</th>
      <td>97.0</td>
      <td>0.413486</td>
      <td>0.270378</td>
      <td>0.000956</td>
      <td>0.209601</td>
      <td>0.412862</td>
      <td>0.616056</td>
      <td>0.952586</td>
    </tr>
    <tr>
      <th>250</th>
      <td>97.0</td>
      <td>0.412703</td>
      <td>0.269847</td>
      <td>0.000955</td>
      <td>0.180196</td>
      <td>0.404551</td>
      <td>0.645144</td>
      <td>0.950033</td>
    </tr>
    <tr>
      <th>100</th>
      <td>97.0</td>
      <td>0.398807</td>
      <td>0.262234</td>
      <td>0.000955</td>
      <td>0.175142</td>
      <td>0.371582</td>
      <td>0.606914</td>
      <td>0.938444</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test Brier score:
metrics_n_estimators.groupby('param_value').describe()[['test_brier_score']].sort_values(('test_brier_score',
                                                                                          'mean'),
                                                                                      ascending=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_brier_score</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>250</th>
      <td>97.0</td>
      <td>0.033402</td>
      <td>0.036566</td>
      <td>0.001986</td>
      <td>0.010507</td>
      <td>0.017989</td>
      <td>0.039544</td>
      <td>0.214301</td>
    </tr>
    <tr>
      <th>100</th>
      <td>97.0</td>
      <td>0.034235</td>
      <td>0.038671</td>
      <td>0.001986</td>
      <td>0.010902</td>
      <td>0.018413</td>
      <td>0.041386</td>
      <td>0.220711</td>
    </tr>
    <tr>
      <th>500</th>
      <td>97.0</td>
      <td>0.035678</td>
      <td>0.042410</td>
      <td>0.001986</td>
      <td>0.010745</td>
      <td>0.019018</td>
      <td>0.039515</td>
      <td>0.224004</td>
    </tr>
    <tr>
      <th>750</th>
      <td>97.0</td>
      <td>0.035756</td>
      <td>0.042796</td>
      <td>0.001986</td>
      <td>0.010051</td>
      <td>0.019100</td>
      <td>0.039533</td>
      <td>0.244093</td>
    </tr>
    <tr>
      <th>1250</th>
      <td>97.0</td>
      <td>0.036437</td>
      <td>0.042925</td>
      <td>0.001986</td>
      <td>0.010723</td>
      <td>0.019099</td>
      <td>0.042245</td>
      <td>0.245623</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>97.0</td>
      <td>0.036851</td>
      <td>0.044559</td>
      <td>0.001986</td>
      <td>0.010861</td>
      <td>0.018987</td>
      <td>0.039517</td>
      <td>0.242675</td>
    </tr>
    <tr>
      <th>1750</th>
      <td>97.0</td>
      <td>0.037360</td>
      <td>0.044541</td>
      <td>0.001986</td>
      <td>0.010660</td>
      <td>0.018545</td>
      <td>0.043360</td>
      <td>0.246554</td>
    </tr>
    <tr>
      <th>1500</th>
      <td>97.0</td>
      <td>0.037628</td>
      <td>0.045930</td>
      <td>0.001986</td>
      <td>0.010622</td>
      <td>0.018489</td>
      <td>0.042427</td>
      <td>0.244613</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>97.0</td>
      <td>0.038011</td>
      <td>0.047029</td>
      <td>0.001986</td>
      <td>0.010393</td>
      <td>0.018520</td>
      <td>0.044183</td>
      <td>0.244061</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test binomial deviance:
metrics_n_estimators.groupby('param_value').describe()[['test_deviance']].sort_values(('test_deviance','mean'),
                                                                                   ascending=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">test_deviance</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>750</th>
      <td>97.0</td>
      <td>2427.744026</td>
      <td>3870.582599</td>
      <td>53.450128</td>
      <td>345.143519</td>
      <td>949.505953</td>
      <td>2685.095464</td>
      <td>22106.981012</td>
    </tr>
    <tr>
      <th>1250</th>
      <td>97.0</td>
      <td>2427.873119</td>
      <td>3870.637555</td>
      <td>53.239442</td>
      <td>345.438257</td>
      <td>951.292076</td>
      <td>2695.459149</td>
      <td>22107.070287</td>
    </tr>
    <tr>
      <th>250</th>
      <td>97.0</td>
      <td>2428.018609</td>
      <td>3871.721799</td>
      <td>53.344754</td>
      <td>343.885114</td>
      <td>949.242762</td>
      <td>2697.410263</td>
      <td>22107.917471</td>
    </tr>
    <tr>
      <th>500</th>
      <td>97.0</td>
      <td>2428.085533</td>
      <td>3870.523374</td>
      <td>53.540951</td>
      <td>344.812311</td>
      <td>948.907512</td>
      <td>2689.600558</td>
      <td>22106.303690</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>97.0</td>
      <td>2428.247280</td>
      <td>3870.311054</td>
      <td>53.388970</td>
      <td>345.243934</td>
      <td>950.717060</td>
      <td>2689.041596</td>
      <td>22106.136324</td>
    </tr>
    <tr>
      <th>1750</th>
      <td>97.0</td>
      <td>2428.392962</td>
      <td>3870.546509</td>
      <td>53.447200</td>
      <td>345.198538</td>
      <td>951.791724</td>
      <td>2701.880355</td>
      <td>22106.648320</td>
    </tr>
    <tr>
      <th>1500</th>
      <td>97.0</td>
      <td>2428.523555</td>
      <td>3870.334408</td>
      <td>53.496438</td>
      <td>345.351901</td>
      <td>952.168705</td>
      <td>2702.105783</td>
      <td>22106.325352</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>97.0</td>
      <td>2429.027810</td>
      <td>3870.648523</td>
      <td>53.500470</td>
      <td>345.194551</td>
      <td>952.198819</td>
      <td>2708.324268</td>
      <td>22107.167220</td>
    </tr>
    <tr>
      <th>100</th>
      <td>97.0</td>
      <td>2430.753893</td>
      <td>3873.716516</td>
      <td>53.437669</td>
      <td>343.369418</td>
      <td>950.084749</td>
      <td>2720.073728</td>
      <td>22110.606157</td>
    </tr>
  </tbody>
</table>
</div>



#### Averages of performance metrics by hyper-parameter value


```python
metrics_n_estimators.groupby('param_value').mean().sort_values('test_roc_auc', ascending=False).drop('store_id',
                                                                                                  axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test_roc_auc</th>
      <th>test_prec_avg</th>
      <th>test_pr_auc</th>
      <th>test_deviance</th>
      <th>test_brier_score</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>0.829800</td>
      <td>0.416903</td>
      <td>0.425950</td>
      <td>2428.085533</td>
      <td>0.035678</td>
    </tr>
    <tr>
      <th>1500</th>
      <td>0.827766</td>
      <td>0.418514</td>
      <td>0.428398</td>
      <td>2428.523555</td>
      <td>0.037628</td>
    </tr>
    <tr>
      <th>750</th>
      <td>0.827545</td>
      <td>0.415493</td>
      <td>0.426481</td>
      <td>2427.744026</td>
      <td>0.035756</td>
    </tr>
    <tr>
      <th>1250</th>
      <td>0.827459</td>
      <td>0.413486</td>
      <td>0.422271</td>
      <td>2427.873119</td>
      <td>0.036437</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>0.827296</td>
      <td>0.415000</td>
      <td>0.424359</td>
      <td>2429.027810</td>
      <td>0.038011</td>
    </tr>
    <tr>
      <th>250</th>
      <td>0.827100</td>
      <td>0.412703</td>
      <td>0.422142</td>
      <td>2428.018609</td>
      <td>0.033402</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>0.826607</td>
      <td>0.415830</td>
      <td>0.420969</td>
      <td>2428.247280</td>
      <td>0.036851</td>
    </tr>
    <tr>
      <th>1750</th>
      <td>0.824530</td>
      <td>0.416144</td>
      <td>0.426932</td>
      <td>2428.392962</td>
      <td>0.037360</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.820743</td>
      <td>0.398807</td>
      <td>0.412879</td>
      <td>2430.753893</td>
      <td>0.034235</td>
    </tr>
  </tbody>
</table>
</div>



#### Frequency of best hyper-parameter values


```python
best_n_estimators_values = metrics_n_estimators.groupby('store_id').idxmax()['test_roc_auc'].values
print('\033[1mRelative frequency of highest performance metric by hyper-parameter value:\033[0m')
print(metrics_n_estimators.reindex(best_n_estimators_values).param_value.value_counts()/len(best_n_estimators_values))
```

    [1mRelative frequency of highest performance metric by hyper-parameter value:[0m
    250.0     0.20
    100.0     0.20
    1500.0    0.11
    500.0     0.11
    1000.0    0.09
    750.0     0.09
    2000.0    0.06
    1250.0    0.06
    1750.0    0.05
    Name: param_value, dtype: float64
    

#### Average performance metric by best hyper-parameter value


```python
# Dataframe with best hyper-parameter value by dataset:
best_n_estimators_values = metrics_n_estimators.reindex(best_n_estimators_values)[['store_id', 'param_value', 'test_roc_auc']]
best_n_estimators_values = best_n_estimators_values.merge(data_info, on='store_id', how='inner')
print('\033[1mShape of best_n_estimators_values:\033[0m ' + str(best_n_estimators_values.shape) + '.')
best_n_estimators_values.head()
```

    [1mShape of best_n_estimators_values:[0m (97, 6).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>param_value</th>
      <th>test_roc_auc</th>
      <th>n_orders</th>
      <th>n_vars</th>
      <th>avg_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>720.0</td>
      <td>500.0</td>
      <td>0.840988</td>
      <td>4028</td>
      <td>1858</td>
      <td>0.011668</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1098.0</td>
      <td>1500.0</td>
      <td>0.971345</td>
      <td>19152</td>
      <td>4026</td>
      <td>0.023705</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1181.0</td>
      <td>750.0</td>
      <td>0.982147</td>
      <td>3467</td>
      <td>2698</td>
      <td>0.033458</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1210.0</td>
      <td>100.0</td>
      <td>0.500000</td>
      <td>4028</td>
      <td>2101</td>
      <td>0.001490</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1241.0</td>
      <td>1750.0</td>
      <td>0.858173</td>
      <td>206</td>
      <td>3791</td>
      <td>0.320388</td>
    </tr>
  </tbody>
</table>
</div>




```python
best_n_estimators_values.groupby('param_value').mean().sort_values('test_roc_auc', ascending=False)[['test_roc_auc']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test_roc_auc</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>750.0</th>
      <td>0.921714</td>
    </tr>
    <tr>
      <th>1500.0</th>
      <td>0.920224</td>
    </tr>
    <tr>
      <th>1250.0</th>
      <td>0.911071</td>
    </tr>
    <tr>
      <th>2000.0</th>
      <td>0.882975</td>
    </tr>
    <tr>
      <th>1750.0</th>
      <td>0.864174</td>
    </tr>
    <tr>
      <th>500.0</th>
      <td>0.857464</td>
    </tr>
    <tr>
      <th>250.0</th>
      <td>0.835418</td>
    </tr>
    <tr>
      <th>1000.0</th>
      <td>0.820608</td>
    </tr>
    <tr>
      <th>100.0</th>
      <td>0.708717</td>
    </tr>
  </tbody>
</table>
</div>



<a id='describing_n_estimators_values'></a>

### Describing hyper-parameter values

#### Average numbers of observations and features by best hyper-parameter value


```python
best_n_estimators_values.groupby('param_value').mean()[['n_orders']].sort_values('n_orders', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_orders</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1000.0</th>
      <td>12530.111111</td>
    </tr>
    <tr>
      <th>500.0</th>
      <td>12116.545455</td>
    </tr>
    <tr>
      <th>1500.0</th>
      <td>11050.272727</td>
    </tr>
    <tr>
      <th>750.0</th>
      <td>10621.111111</td>
    </tr>
    <tr>
      <th>2000.0</th>
      <td>6896.500000</td>
    </tr>
    <tr>
      <th>1750.0</th>
      <td>5938.600000</td>
    </tr>
    <tr>
      <th>250.0</th>
      <td>4339.850000</td>
    </tr>
    <tr>
      <th>1250.0</th>
      <td>2864.166667</td>
    </tr>
    <tr>
      <th>100.0</th>
      <td>2473.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
best_n_estimators_values.groupby('param_value').mean()[['n_vars']].sort_values('n_vars', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_vars</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1750.0</th>
      <td>3015.800000</td>
    </tr>
    <tr>
      <th>1500.0</th>
      <td>2458.727273</td>
    </tr>
    <tr>
      <th>750.0</th>
      <td>2448.333333</td>
    </tr>
    <tr>
      <th>500.0</th>
      <td>2384.636364</td>
    </tr>
    <tr>
      <th>1000.0</th>
      <td>2353.777778</td>
    </tr>
    <tr>
      <th>2000.0</th>
      <td>2345.833333</td>
    </tr>
    <tr>
      <th>1250.0</th>
      <td>2333.666667</td>
    </tr>
    <tr>
      <th>250.0</th>
      <td>2283.450000</td>
    </tr>
    <tr>
      <th>100.0</th>
      <td>2203.800000</td>
    </tr>
  </tbody>
</table>
</div>



#### Average of response variable by best hyper-parameter value


```python
best_n_estimators_values.groupby('param_value').mean()[['avg_y']].sort_values('avg_y', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_y</th>
    </tr>
    <tr>
      <th>param_value</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1750.0</th>
      <td>0.092578</td>
    </tr>
    <tr>
      <th>250.0</th>
      <td>0.072071</td>
    </tr>
    <tr>
      <th>1250.0</th>
      <td>0.065581</td>
    </tr>
    <tr>
      <th>2000.0</th>
      <td>0.049240</td>
    </tr>
    <tr>
      <th>500.0</th>
      <td>0.047090</td>
    </tr>
    <tr>
      <th>100.0</th>
      <td>0.040656</td>
    </tr>
    <tr>
      <th>750.0</th>
      <td>0.035418</td>
    </tr>
    <tr>
      <th>1500.0</th>
      <td>0.029324</td>
    </tr>
    <tr>
      <th>1000.0</th>
      <td>0.025724</td>
    </tr>
  </tbody>
</table>
</div>



#### Most frequent best hyper-parameter values by quartile of number of observations


```python
best_n_estimators_values['quartile_n_orders'] = percentile_cut(best_n_estimators_values.n_orders, p=4)['percentile']

print('\033[1mFrequency of best hyper-parameter values by quartile of number of observations:\033[0m')
for q in range(1,5):
    print('\033[1mNumber of orders in ' +
          str(np.sort(np.unique(percentile_cut(best_n_estimators_values.n_orders, p=4)['interval']))[q-1]) +
          ' (quartile ' + str(q) + ')\033[0m:')
    print(best_n_estimators_values[best_n_estimators_values.quartile_n_orders==q].param_value.value_counts())
    print('\n')
```

    [1mFrequency of best hyper-parameter values by quartile of number of observations:[0m
    [1mNumber of orders in (156.998, 999.0] (quartile 1)[0m:
    100.0     8
    250.0     7
    1750.0    3
    2000.0    2
    1500.0    1
    500.0     1
    750.0     1
    1000.0    1
    1250.0    1
    Name: param_value, dtype: int64
    
    
    [1mNumber of orders in (999.0, 2812.0] (quartile 2)[0m:
    250.0     8
    100.0     6
    1250.0    4
    1000.0    2
    2000.0    2
    1500.0    1
    500.0     1
    Name: param_value, dtype: int64
    
    
    [1mNumber of orders in (2812.0, 7946.0] (quartile 3)[0m:
    100.0     5
    250.0     4
    1500.0    4
    750.0     4
    500.0     3
    1000.0    2
    1250.0    1
    2000.0    1
    Name: param_value, dtype: int64
    
    
    [1mNumber of orders in (7946.0, 63963.001] (quartile 4)[0m:
    500.0     6
    1500.0    5
    750.0     4
    1000.0    4
    1750.0    2
    100.0     1
    250.0     1
    2000.0    1
    Name: param_value, dtype: int64
    
    
    

#### Most frequent best hyper-parameter values by quartile of number of features


```python
best_n_estimators_values['quartile_n_vars'] = percentile_cut(best_n_estimators_values.n_vars, p=4)['percentile']

print('\033[1mFrequency of best hyper-parameter values by quartile of number of features:\033[0m')
for q in range(1,5):
    print('\033[1mNumber of vars in ' +
          str(np.sort(np.unique(percentile_cut(best_n_estimators_values.n_vars, p=4)['interval']))[q-1]) +
          ' (quartile ' + str(q) + ')\033[0m:')
    print(best_n_estimators_values[best_n_estimators_values.quartile_n_vars==q].param_value.value_counts())
    print('\n')
```

    [1mFrequency of best hyper-parameter values by quartile of number of features:[0m
    [1mNumber of vars in (1415.998, 2069.0] (quartile 1)[0m:
    100.0     8
    250.0     6
    1000.0    3
    750.0     2
    1500.0    2
    500.0     2
    1250.0    1
    2000.0    1
    Name: param_value, dtype: int64
    
    
    [1mNumber of vars in (2069.0, 2321.0] (quartile 2)[0m:
    100.0     7
    250.0     5
    1500.0    4
    500.0     2
    750.0     2
    2000.0    2
    1250.0    2
    Name: param_value, dtype: int64
    
    
    [1mNumber of vars in (2321.0, 2534.0] (quartile 3)[0m:
    250.0     5
    500.0     4
    1000.0    3
    1500.0    3
    750.0     3
    1250.0    2
    2000.0    2
    1750.0    1
    100.0     1
    Name: param_value, dtype: int64
    
    
    [1mNumber of vars in (2534.0, 4026.001] (quartile 4)[0m:
    100.0     4
    250.0     4
    1750.0    4
    500.0     3
    1000.0    3
    750.0     2
    1500.0    2
    2000.0    1
    1250.0    1
    Name: param_value, dtype: int64
    
    
    

#### Correlation between performance metric of best hyper-parameter value and dataset information


```python
best_n_estimators_values[['test_roc_auc', 'n_orders', 'n_vars', 'avg_y']].corr()[['test_roc_auc']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test_roc_auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>test_roc_auc</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>n_orders</th>
      <td>0.295579</td>
    </tr>
    <tr>
      <th>n_vars</th>
      <td>0.041541</td>
    </tr>
    <tr>
      <th>avg_y</th>
      <td>0.165359</td>
    </tr>
  </tbody>
</table>
</div>



#### Correlation between performance metric and dataset information by hyper-parameter value


```python
metrics_n_estimators = metrics_n_estimators.merge(data_info, on='store_id', how='left')
print('\033[1mShape of metrics_n_estimators:\033[0m ' + str(metrics_n_estimators.shape) + '.')
metrics_n_estimators.head()
```

    [1mShape of metrics_n_estimators:[0m (900, 10).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>param_value</th>
      <th>test_roc_auc</th>
      <th>test_prec_avg</th>
      <th>test_pr_auc</th>
      <th>test_deviance</th>
      <th>test_brier_score</th>
      <th>n_orders</th>
      <th>n_vars</th>
      <th>avg_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11729</td>
      <td>100</td>
      <td>0.801450</td>
      <td>0.165249</td>
      <td>0.157897</td>
      <td>885.405667</td>
      <td>0.041386</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11729</td>
      <td>250</td>
      <td>0.771918</td>
      <td>0.140431</td>
      <td>0.131739</td>
      <td>885.743592</td>
      <td>0.044329</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11729</td>
      <td>500</td>
      <td>0.772637</td>
      <td>0.180357</td>
      <td>0.173761</td>
      <td>886.097185</td>
      <td>0.041737</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11729</td>
      <td>750</td>
      <td>0.749840</td>
      <td>0.195791</td>
      <td>0.191797</td>
      <td>886.435896</td>
      <td>0.042187</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11729</td>
      <td>1000</td>
      <td>0.739582</td>
      <td>0.178299</td>
      <td>0.173366</td>
      <td>886.065753</td>
      <td>0.045934</td>
      <td>2570</td>
      <td>1596</td>
      <td>0.037354</td>
    </tr>
  </tbody>
</table>
</div>




```python
for v in metrics_n_estimators.param_value.unique():
    print('\033[1mn_estimators = ' + str(v) + '\033[0m')
    print(metrics_n_estimators[metrics_n_estimators.param_value==v][['test_roc_auc',
                                                               'n_orders', 'n_vars',
                                                               'avg_y']].corr()[['test_roc_auc']])
    print('\n')
```

    [1mn_estimators = 100[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.298490
    n_vars            0.048438
    avg_y             0.189709
    
    
    [1mn_estimators = 250[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.301489
    n_vars            0.048528
    avg_y             0.177903
    
    
    [1mn_estimators = 500[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.305460
    n_vars            0.054179
    avg_y             0.161834
    
    
    [1mn_estimators = 750[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.304751
    n_vars            0.056807
    avg_y             0.160490
    
    
    [1mn_estimators = 1000[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.309234
    n_vars            0.061954
    avg_y             0.160907
    
    
    [1mn_estimators = 1250[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.310433
    n_vars            0.068218
    avg_y             0.158200
    
    
    [1mn_estimators = 1500[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.309599
    n_vars            0.071022
    avg_y             0.152438
    
    
    [1mn_estimators = 1750[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.315503
    n_vars            0.069494
    avg_y             0.162369
    
    
    [1mn_estimators = 2000[0m
                  test_roc_auc
    test_roc_auc      1.000000
    n_orders          0.309337
    n_vars            0.070663
    avg_y             0.154681
    
    
    

<a id='data_vis_n_estimators'></a>

### Data visualization

#### Average of performance metric by hyper-parameter value


```python
# Select a performance metric:
metric = 'test_roc_auc'

fig=px.scatter(x=metrics_n_estimators['param_value'].apply(lambda x: 'M = ' + str(x)).unique(),
               y=metrics_n_estimators.groupby('param_value').mean()[metric], 
               error_y=np.array(metrics_n_estimators.groupby('param_value').std()['test_roc_auc']),
               color_discrete_sequence=['#0b6fab'],
               width=900, height=500,
               title='Average of ' + metric + ' by number of estimators',
               labels={'y': metric, 'x': ''})

fig.add_trace(
    go.Scatter(
        x=metrics_n_estimators['param_value'].apply(lambda x: 'M = ' + str(x)).unique(),
        y=metrics_n_estimators.groupby('param_value').mean()[metric],
        line = dict(color='#0b6fab', width=2, dash='dash'),
        name='avg_' + metric
              )
)
```


<div>                            <div id="6cb54dea-4706-42be-a3ee-fcb2295308dc" class="plotly-graph-div" style="height:500px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("6cb54dea-4706-42be-a3ee-fcb2295308dc")) {                    Plotly.newPlot(                        "6cb54dea-4706-42be-a3ee-fcb2295308dc",                        [{"error_y": {"array": [0.15024915839959235, 0.15528503874067434, 0.15401811285537245, 0.15723248036080412, 0.15651221307396743, 0.1551441733710658, 0.15518285729319967, 0.15761782954125322, 0.15477105297908958]}, "hovertemplate": "=%{x}<br>test_roc_auc=%{y}<extra></extra>", "legendgroup": "", "marker": {"color": "#0b6fab", "symbol": "circle"}, "mode": "markers", "name": "", "orientation": "v", "showlegend": false, "type": "scatter", "x": ["M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000"], "xaxis": "x", "y": [0.820743019034623, 0.8271002249603635, 0.8298003950429392, 0.8275451288448223, 0.8266072928789466, 0.8274589449863505, 0.8277660221918464, 0.8245304870519805, 0.8272955691589773], "yaxis": "y"}, {"line": {"color": "#0b6fab", "dash": "dash", "width": 2}, "name": "avg_test_roc_auc", "type": "scatter", "x": ["M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000"], "y": [0.820743019034623, 0.8271002249603635, 0.8298003950429392, 0.8275451288448223, 0.8266072928789466, 0.8274589449863505, 0.8277660221918464, 0.8245304870519805, 0.8272955691589773]}],                        {"height": 500, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Average of test_roc_auc by number of estimators"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": ""}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "test_roc_auc"}}},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('6cb54dea-4706-42be-a3ee-fcb2295308dc');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


<a id='reference20'></a>
#### Boxplot of performance metric by hyper-parameter value


```python
# Select a performance metric:
metric = 'test_roc_auc'

px.box(data_frame=metrics_n_estimators,
       x=metrics_n_estimators['param_value'].apply(lambda x: 'M = ' + str(x)),
       y=metric, hover_data=['store_id'],
       color_discrete_sequence=['#0b6fab'],
       width=900, height=500,
       labels={'x': ' '},
       title='Distribution of ' + metric + ' by number of estimators.')
```


<div>                            <div id="c81c2c27-9cc2-46ca-9d7f-54d3b45d6f4a" class="plotly-graph-div" style="height:500px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("c81c2c27-9cc2-46ca-9d7f-54d3b45d6f4a")) {                    Plotly.newPlot(                        "c81c2c27-9cc2-46ca-9d7f-54d3b45d6f4a",                        [{"alignmentgroup": "True", "customdata": [[11729], [11729], [11729], [11729], [11729], [11729], [11729], [11729], [11729], [10311], [10311], [10311], [10311], [10311], [10311], [10311], [10311], [10311], [7988], [7988], [7988], [7988], [7988], [7988], [7988], [7988], [7988], [4736], [4736], [4736], [4736], [4736], [4736], [4736], [4736], [4736], [3481], [3481], [3481], [3481], [3481], [3481], [3481], [3481], [3481], [4838], [4838], [4838], [4838], [4838], [4838], [4838], [4838], [4838], [5848], [5848], [5848], [5848], [5848], [5848], [5848], [5848], [5848], [6106], [6106], [6106], [6106], [6106], [6106], [6106], [6106], [6106], [1559], [1559], [1559], [1559], [1559], [1559], [1559], [1559], [1559], [5342], [5342], [5342], [5342], [5342], [5342], [5342], [5342], [5342], [3781], [3781], [3781], [3781], [3781], [3781], [3781], [3781], [3781], [4408], [4408], [4408], [4408], [4408], [4408], [4408], [4408], [4408], [7292], [7292], [7292], [7292], [7292], [7292], [7292], [7292], [7292], [6044], [6044], [6044], [6044], [6044], [6044], [6044], [6044], [6044], [8181], [8181], [8181], [8181], [8181], [8181], [8181], [8181], [8181], [2352], [2352], [2352], [2352], [2352], [2352], [2352], [2352], [2352], [9491], [9491], [9491], [9491], [9491], [9491], [9491], [9491], [9491], [5847], [5847], [5847], [5847], [5847], [5847], [5847], [5847], [5847], [7939], [7939], [7939], [7939], [7939], [7939], [7939], [7939], [7939], [6078], [6078], [6078], [6078], [6078], [6078], [6078], [6078], [6078], [10268], [10268], [10268], [10268], [10268], [10268], [10268], [10268], [10268], [10060], [10060], [10060], [10060], [10060], [10060], [10060], [10060], [10060], [6256], [6256], [6256], [6256], [6256], [6256], [6256], [6256], [6256], [8436], [8436], [8436], [8436], [8436], [8436], [8436], [8436], [8436], [5085], [5085], [5085], [5085], [5085], [5085], [5085], [5085], [5085], [8783], [8783], [8783], [8783], [8783], [8783], [8783], [8783], [8783], [6047], [6047], [6047], [6047], [6047], [6047], [6047], [6047], [6047], [8832], [8832], [8832], [8832], [8832], [8832], [8832], [8832], [8832], [1961], [1961], [1961], [1961], [1961], [1961], [1961], [1961], [1961], [7845], [7845], [7845], [7845], [7845], [7845], [7845], [7845], [7845], [2699], [2699], [2699], [2699], [2699], [2699], [2699], [2699], [2699], [6004], [6004], [6004], [6004], [6004], [6004], [6004], [6004], [6004], [2868], [2868], [2868], [2868], [2868], [2868], [2868], [2868], [2868], [1875], [1875], [1875], [1875], [1875], [1875], [1875], [1875], [1875], [5593], [5593], [5593], [5593], [5593], [5593], [5593], [5593], [5593], [7849], [7849], [7849], [7849], [7849], [7849], [7849], [7849], [7849], [11223], [11223], [11223], [11223], [11223], [11223], [11223], [11223], [11223], [6170], [6170], [6170], [6170], [6170], [6170], [6170], [6170], [6170], [5168], [5168], [5168], [5168], [5168], [5168], [5168], [5168], [5168], [2866], [2866], [2866], [2866], [2866], [2866], [2866], [2866], [2866], [3437], [3437], [3437], [3437], [3437], [3437], [3437], [3437], [3437], [6929], [6929], [6929], [6929], [6929], [6929], [6929], [6929], [6929], [8894], [8894], [8894], [8894], [8894], [8894], [8894], [8894], [8894], [9177], [9177], [9177], [9177], [9177], [9177], [9177], [9177], [9177], [10349], [10349], [10349], [10349], [10349], [10349], [10349], [10349], [10349], [3988], [3988], [3988], [3988], [3988], [3988], [3988], [3988], [3988], [1549], [1549], [1549], [1549], [1549], [1549], [1549], [1549], [1549], [9541], [9541], [9541], [9541], [9541], [9541], [9541], [9541], [9541], [1181], [1181], [1181], [1181], [1181], [1181], [1181], [1181], [1181], [7790], [7790], [7790], [7790], [7790], [7790], [7790], [7790], [7790], [5663], [5663], [5663], [5663], [5663], [5663], [5663], [5663], [5663], [4601], [4601], [4601], [4601], [4601], [4601], [4601], [4601], [4601], [1603], [1603], [1603], [1603], [1603], [1603], [1603], [1603], [1603], [2212], [2212], [2212], [2212], [2212], [2212], [2212], [2212], [2212], [9761], [9761], [9761], [9761], [9761], [9761], [9761], [9761], [9761], [5428], [5428], [5428], [5428], [5428], [5428], [5428], [5428], [5428], [1098], [1098], [1098], [1098], [1098], [1098], [1098], [1098], [1098], [5939], [5939], [5939], [5939], [5939], [5939], [5939], [5939], [5939], [7333], [7333], [7333], [7333], [7333], [7333], [7333], [7333], [7333], [8358], [8358], [8358], [8358], [8358], [8358], [8358], [8358], [8358], [10650], [10650], [10650], [10650], [10650], [10650], [10650], [10650], [10650], [6083], [6083], [6083], [6083], [6083], [6083], [6083], [6083], [6083], [1424], [1424], [1424], [1424], [1424], [1424], [1424], [1424], [1424], [9281], [9281], [9281], [9281], [9281], [9281], [9281], [9281], [9281], [7161], [7161], [7161], [7161], [7161], [7161], [7161], [7161], [7161], [7185], [7185], [7185], [7185], [7185], [7185], [7185], [7185], [7185], [12980], [12980], [12980], [12980], [12980], [12980], [12980], [12980], [12980], [8282], [8282], [8282], [8282], [8282], [8282], [8282], [8282], [8282], [3962], [3962], [3962], [3962], [3962], [3962], [3962], [3962], [3962], [720], [720], [720], [720], [720], [720], [720], [720], [720], [11723], [11723], [11723], [11723], [11723], [11723], [11723], [11723], [11723], [8446], [8446], [8446], [8446], [8446], [8446], [8446], [8446], [8446], [8790], [8790], [8790], [8790], [8790], [8790], [8790], [8790], [8790], [1241], [1241], [1241], [1241], [1241], [1241], [1241], [1241], [1241], [9098], [9098], [9098], [9098], [9098], [9098], [9098], [9098], [9098], [1739], [1739], [1739], [1739], [1739], [1739], [1739], [1739], [1739], [4636], [4636], [4636], [4636], [4636], [4636], [4636], [4636], [4636], [8421], [8421], [8421], [8421], [8421], [8421], [8421], [8421], [8421], [5215], [5215], [5215], [5215], [5215], [5215], [5215], [5215], [5215], [7062], [7062], [7062], [7062], [7062], [7062], [7062], [7062], [7062], [6714], [6714], [6714], [6714], [6714], [6714], [6714], [6714], [6714], [7630], [7630], [7630], [7630], [7630], [7630], [7630], [7630], [7630], [6970], [6970], [6970], [6970], [6970], [6970], [6970], [6970], [6970], [3146], [3146], [3146], [3146], [3146], [3146], [3146], [3146], [3146], [5860], [5860], [5860], [5860], [5860], [5860], [5860], [5860], [5860], [7755], [7755], [7755], [7755], [7755], [7755], [7755], [7755], [7755], [6105], [6105], [6105], [6105], [6105], [6105], [6105], [6105], [6105], [4030], [4030], [4030], [4030], [4030], [4030], [4030], [4030], [4030], [3859], [3859], [3859], [3859], [3859], [3859], [3859], [3859], [3859], [4268], [4268], [4268], [4268], [4268], [4268], [4268], [4268], [4268], [9409], [9409], [9409], [9409], [9409], [9409], [9409], [9409], [9409], [1979], [1979], [1979], [1979], [1979], [1979], [1979], [1979], [1979], [12658], [12658], [12658], [12658], [12658], [12658], [12658], [12658], [12658], [5394], [5394], [5394], [5394], [5394], [5394], [5394], [5394], [5394], [1210], [1210], [1210], [1210], [1210], [1210], [1210], [1210], [1210], [6971], [6971], [6971], [6971], [6971], [6971], [6971], [6971], [6971], [2056], [2056], [2056], [2056], [2056], [2056], [2056], [2056], [2056], [2782], [2782], [2782], [2782], [2782], [2782], [2782], [2782], [2782], [4974], [4974], [4974], [4974], [4974], [4974], [4974], [4974], [4974], [6966], [6966], [6966], [6966], [6966], [6966], [6966], [6966], [6966]], "hovertemplate": " =%{x}<br>test_roc_auc=%{y}<br>store_id=%{customdata[0]}<extra></extra>", "legendgroup": "", "marker": {"color": "#0b6fab"}, "name": "", "notched": false, "offsetgroup": "", "orientation": "v", "showlegend": false, "type": "box", "x": ["M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000", "M = 100", "M = 250", "M = 500", "M = 750", "M = 1000", "M = 1250", "M = 1500", "M = 1750", "M = 2000"], "x0": " ", "xaxis": "x", "y": [0.8014500755550389, 0.7719182262001627, 0.7726374520516099, 0.7498401720330117, 0.7395821225154016, 0.7373300011623851, 0.7334796001394862, 0.7157241659886087, 0.7243112867604323, 0.8774571572580644, 0.755859375, 0.8685735887096775, 0.8806703629032259, 0.8523815524193549, 0.8728578629032259, 0.8817414314516129, 0.7521421370967742, 0.8676285282258065, 0.5774793388429752, 0.5783976124885215, 0.5578512396694215, 0.5443985307621672, 0.5454545454545454, 0.5393480257116621, 0.5367309458218549, 0.5373737373737373, 0.5393480257116621, 0.8404525386313466, 0.8811810154525387, 0.8882450331125826, 0.8947019867549669, 0.8961368653421633, 0.8942604856512142, 0.8942604856512142, 0.8947019867549668, 0.8944812362030905, 0.7663135267925685, 0.7865295050923794, 0.7853779620246686, 0.7825118992783663, 0.7869261477045908, 0.7757433850248221, 0.785134858488152, 0.7771764163979733, 0.7854419366395415, 0.9673275298654218, 0.9687639860989511, 0.9723669865319987, 0.9741471245124587, 0.9743088827006509, 0.9738217221661671, 0.9737726869485805, 0.9740484737788526, 0.9737812463504669, 0.9672473867595819, 0.9730545876887341, 0.9681765389082462, 0.967479674796748, 0.9707317073170733, 0.967479674796748, 0.9721254355400697, 0.9688734030197445, 0.9693379790940766, 0.7122153209109732, 0.7122153209109732, 0.7122153209109732, 0.7142857142857143, 0.7122153209109732, 0.7122153209109732, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.9096450617283951, 0.9324074074074074, 0.9239197530864197, 0.9290123456790123, 0.9229938271604938, 0.933179012345679, 0.9277777777777778, 0.9316358024691358, 0.9325617283950617, 0.9277882329077407, 0.9374739404320536, 0.9345918930797056, 0.9360540461352955, 0.9345552688220512, 0.93474684186209, 0.9359300871093883, 0.935992066622342, 0.9360315081305851, 0.9564679645745484, 0.9666423148729648, 0.9668616234244317, 0.9713061817953965, 0.9716408409486093, 0.971283396491348, 0.9719071441896762, 0.9701241941478795, 0.9702110631195645, 0.7427083333333333, 0.76484375, 0.7635416666666667, 0.76328125, 0.76171875, 0.7645833333333334, 0.7645833333333333, 0.7614583333333333, 0.7619791666666667, 0.49410377358490565, 0.49410377358490565, 0.49410377358490565, 0.49410377358490565, 0.49410377358490565, 0.49410377358490565, 0.49410377358490565, 0.49410377358490565, 0.49410377358490565, 0.9577313819037812, 0.9659016898984545, 0.9711076854125495, 0.9711287575207963, 0.9705095421953313, 0.9710455692380393, 0.9738321651430433, 0.973776142108267, 0.9730656143137331, 0.48984657039711194, 0.48984657039711194, 0.48894404332129965, 0.48894404332129965, 0.48894404332129965, 0.47405234657039713, 0.47405234657039713, 0.47405234657039713, 0.48894404332129965, 0.8549127976746048, 0.9176678044747859, 0.8879836795647884, 0.8969155844155843, 0.8864369716525775, 0.9218095815888423, 0.9226362703005413, 0.9111442971812582, 0.9090759086908984, 0.9403931039381426, 0.9469012031239958, 0.9506790192357779, 0.9497420910748291, 0.9497529147936559, 0.9476334953508745, 0.9462081468778646, 0.9459334950126334, 0.9439317835120937, 0.9615267421462493, 0.969901754660886, 0.9729618553873878, 0.9737910860963475, 0.9735669108653164, 0.9730641489394118, 0.9723829174121037, 0.9717190975532253, 0.9727594447418937, 0.9351963141025641, 0.9284071906354514, 0.921875, 0.9118763935340023, 0.9105264074693422, 0.912320582497213, 0.908139980490524, 0.9060148411371237, 0.9113538182831661, 0.4822222222222222, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.5379704301075269, 0.5372983870967742, 0.5379704301075269, 0.5379704301075269, 0.5372983870967742, 0.5379704301075269, 0.5379704301075269, 0.5379704301075269, 0.5379704301075269, 0.8696461824953445, 0.8771828115667053, 0.8745651944766523, 0.872597589684129, 0.8725624538842627, 0.8701029478936086, 0.8696461824953446, 0.870629984891606, 0.8709110712905379, 0.9256564525908306, 0.9242217291748562, 0.9270367631218337, 0.9281272417553375, 0.9236531700502597, 0.9207370836026216, 0.9212712182489712, 0.9425644237453388, 0.9235387919451162, 0.6729127767404641, 0.6895172045878901, 0.6807148572952787, 0.6892504667911443, 0.6875166711122966, 0.6907175246732462, 0.7013870365430781, 0.6937850093358229, 0.6735129367831422, 0.7354761904761905, 0.6902380952380952, 0.685, 0.6833333333333333, 0.6861904761904762, 0.6861904761904762, 0.687142857142857, 0.6823809523809523, 0.6838095238095239, 0.8973077855930147, 0.8911876465356939, 0.8714689950683159, 0.8617673215296305, 0.8557361144797476, 0.8525507316678792, 0.8532783571832806, 0.8536987630366238, 0.8540706605222733, 0.9203308194231734, 0.9275403469710815, 0.9258511795214991, 0.9292351449788291, 0.9275636735691948, 0.9229980952626093, 0.9249108763078983, 0.9247709167192185, 0.9229208761791998, 0.9175340721881085, 0.9244421147221805, 0.923318855773551, 0.9140706904298338, 0.91594278867755, 0.9143889471319455, 0.9169911636962708, 0.9169911636962708, 0.9161674404672757, 0.9189055429864253, 0.9206495098039216, 0.9155354449472097, 0.9133437028657617, 0.9130137631975869, 0.9121417797888387, 0.9127309577677224, 0.9127780920060332, 0.9129901960784313, 0.72140522875817, 0.7294730392156864, 0.7276348039215687, 0.7282475490196079, 0.728656045751634, 0.7279411764705883, 0.7285539215686275, 0.7285539215686275, 0.7280433006535948, 0.9731195566112432, 0.9744435647048474, 0.9696181930148675, 0.9674100466261987, 0.9667766341163015, 0.9666226796868127, 0.9667238497404769, 0.9661608163983461, 0.966314770827835, 0.8684734984252752, 0.8719822809502097, 0.8839006670310325, 0.8851734310880548, 0.8906736439702991, 0.889175919243649, 0.8920113295628516, 0.8880813119585469, 0.887308292744792, 0.9503746177370032, 0.9792278287461773, 0.9786620795107033, 0.9770412844036697, 0.9767048929663609, 0.9790443425076453, 0.9799159021406727, 0.9775764525993882, 0.978448012232416, 0.9549100156494523, 0.9595803990610329, 0.943955399061033, 0.9392361111111112, 0.9391627543035994, 0.9395295383411582, 0.9442488262910799, 0.9396028951486698, 0.9398474178403756, 0.9297978336802738, 0.9359343753307695, 0.9395396394171401, 0.9398820167237061, 0.9392765762269343, 0.939718448999753, 0.9388562960872172, 0.9387127685848357, 0.9382927707017606, 0.8651880248265791, 0.880878887450629, 0.872348899731156, 0.8645657008198082, 0.8613213183311759, 0.8639350791596137, 0.8549736134621129, 0.8551976501045504, 0.8613130206777523, 0.7101544769016765, 0.7163408741489128, 0.7089830880737975, 0.692949703492203, 0.6869463357493228, 0.6791126729628817, 0.6695951387363643, 0.676843107108866, 0.6769895307123509, 0.9723194152865253, 0.9763213892974965, 0.9777860292705839, 0.9774940533712901, 0.9773591732004209, 0.9767307902867235, 0.9765197424899514, 0.9768640836320532, 0.9763198024719568, 0.5777972027972027, 0.5358391608391608, 0.5305944055944056, 0.5340909090909092, 0.5375874125874126, 0.5305944055944056, 0.5305944055944056, 0.5305944055944056, 0.5358391608391608, 0.9392242885076922, 0.9439785637084395, 0.9448112511215628, 0.9442863718804321, 0.9436295748513045, 0.9425871451350429, 0.9409364278407956, 0.9417436390673481, 0.9390323446362682, 0.9399355877616747, 0.9446054750402576, 0.9473429951690822, 0.9489533011272141, 0.946537842190016, 0.9508856682769726, 0.9458937198067633, 0.9518518518518518, 0.9446054750402577, 0.8947845804988662, 0.9276946334089193, 0.9238851095993953, 0.924368858654573, 0.9239758125472411, 0.9247165532879819, 0.9250793650793652, 0.9184126984126985, 0.9255026455026456, 0.847059276811343, 0.8940260841087289, 0.8836232637885532, 0.8956057716388295, 0.8962607640293591, 0.9055077154250708, 0.8841723015276735, 0.8946714442582211, 0.8780558284690516, 0.9279289035201709, 0.9333903664558607, 0.9351699442663663, 0.9341520113199919, 0.9400719050506801, 0.936750989055416, 0.9404545323283953, 0.9399563949291057, 0.937436830402264, 0.7651915930963848, 0.7989372868013644, 0.791173259071787, 0.7920535631152605, 0.7971004985568091, 0.7994388061722857, 0.7902633294114659, 0.7895121084128287, 0.7898401063136422, 0.9769491129785248, 0.9873482726423904, 0.9875816993464052, 0.9875583566760037, 0.9874883286647993, 0.9877684407096172, 0.9877450980392157, 0.9876050420168068, 0.9874883286647993, null, null, null, null, null, null, null, null, null, 0.9163302575932333, 0.9329632833525567, 0.949115724721261, 0.9563965782391388, 0.948558246828143, 0.947822952710496, 0.9468713956170703, 0.9485438292964244, 0.9469819300269127, 0.9704646564508156, 0.9760919073511758, 0.9804877833486336, 0.982147270672975, 0.9812204293482099, 0.9810968505049078, 0.9816176470588235, 0.9819927971188476, 0.9810615422639644, 0.7034952606635071, 0.7005331753554503, 0.6990521327014219, 0.7005331753554503, 0.7008293838862559, 0.6994470774091628, 0.7001382306477094, 0.7001382306477094, 0.6988546603475514, 0.9400704659384636, 0.9514857768497973, 0.9543037948167823, 0.9534557193683084, 0.9530901721494324, 0.9527176916297947, 0.951954143443797, 0.9507803503765667, 0.9500047057934955, 0.9729511238337575, 0.9682949886909811, 0.9681050325134295, 0.968352417302799, 0.968158043539723, 0.9663600862312695, 0.9684142635001414, 0.9681845490528698, 0.9663689214023184, 0.9662613284061733, 0.9721502339834238, 0.9730366457526861, 0.9741010011127673, 0.9735892509318718, 0.9735739007880055, 0.9734310263720194, 0.9728559863671831, 0.9722724447442052, 0.9133195244956772, 0.8999009365994237, 0.9346271613832853, 0.931772334293948, 0.912797190201729, 0.8936329250720461, 0.9287283861671469, 0.9119326368876081, 0.9311419308357349, 0.6554793300071277, 0.6884818246614397, 0.6785317177476836, 0.6615555951532431, 0.6693317890235211, 0.6890573770491802, 0.6747148966500357, 0.6984283677833214, 0.6671881682109766, 0.8801467108376716, 0.9099513832121499, 0.9133610119175666, 0.9140493912145592, 0.9049929010884996, 0.913834272684249, 0.9141784623327454, 0.9136836897130318, 0.9047132469990965, 0.9436782179445837, 0.9598626855963199, 0.9641576489367525, 0.9643824769590652, 0.9679852377369307, 0.9702220993739858, 0.9713448613803338, 0.9709424546573854, 0.97090229273396, 0.9785649819494585, 0.9880415162454874, 0.9821750902527075, 0.9871389891696751, 0.9943592057761733, 0.9812725631768954, 0.9808212996389892, 0.9853339350180506, 0.9853339350180506, 0.9437620326378848, 0.9542011980995663, 0.9552836190869656, 0.9530543276182607, 0.9534311092749432, 0.9526197066721752, 0.9510613509605453, 0.9508828754389588, 0.9501541003924808, 0.8874075009072405, 0.8977145426718629, 0.8993219362880449, 0.906270215686584, 0.8987065905110525, 0.8911725492671074, 0.8910976033070891, 0.9060927120970669, 0.9053452247589896, 0.8625341255821424, 0.8561506343343503, 0.8469300358653177, 0.8477597559017184, 0.8455783951608586, 0.8462073764787753, 0.8463010545474012, 0.8459531074353621, 0.8477999036454151, 0.6138248482203088, 0.61313043133209, 0.6118011190032142, 0.612981627713186, 0.6296079520653943, 0.6122673703424468, 0.6286655291456689, 0.6124260942026111, 0.6293103448275862, 0.9487955988917779, 0.9570903216117285, 0.9575599079299923, 0.9576486518896047, 0.9577523647098747, 0.9572400513415471, 0.9564807517723707, 0.9562273503455255, 0.9559037104186724, 0.88374545220387, 0.8482189810790216, 0.8642422486235875, 0.8438402180795689, 0.8260192966075319, 0.831852281142342, 0.8069695311074623, 0.8132961997059359, 0.8293999592174035, 0.8123486682808717, 0.7859160613397901, 0.8137610976594027, 0.8048829701372074, 0.812953995157385, 0.8109362389023406, 0.7830912025827279, 0.8036723163841808, 0.7909604519774011, 0.7711291098387874, 0.764565780694813, 0.7670444847864203, 0.7649226520194262, 0.7666139117752021, 0.7655898462350075, 0.7678358081583888, 0.7667070086424925, 0.765531660692951, 0.9067276887871852, 0.9070022883295195, 0.9067048054919908, 0.9064759725400457, 0.9056292906178489, 0.9065675057208238, 0.9051258581235698, 0.9056292906178489, 0.9055148741418764, 0.8978047313552526, 0.9106188184977279, 0.917669072440524, 0.9139100507885591, 0.915062817428495, 0.9166499599037691, 0.9193731622560813, 0.9153802459235498, 0.921678695535953, 0.9522267471563255, 0.9676813626931563, 0.9640308147189746, 0.9646722194097653, 0.9673083955925132, 0.9669975654786646, 0.9668346736581424, 0.9669378965497568, 0.9673985522953156, 0.7927382542863505, 0.8408901135604543, 0.8409875306167891, 0.8367985971943889, 0.8345719216210199, 0.8398324426631041, 0.84083444667112, 0.839108773101759, 0.8352399242930306, 0.7070959010054136, 0.8875241686001546, 0.9178364269141531, 0.9193155452436195, 0.9139017788089714, 0.8897090100541377, 0.9133410672853829, 0.8877465197215777, 0.8879737045630317, 0.49235181644359466, 0.49235181644359466, 0.4933078393881453, 0.4933078393881453, 0.49713193116634796, 0.4933078393881453, 0.49713193116634796, 0.49713193116634796, 0.4933078393881453, 0.8008518062397372, 0.8008004926108374, 0.8037766830870279, 0.8071120689655172, 0.7998255336617406, 0.799055829228243, 0.8124486863711001, 0.807471264367816, 0.8210693760262725, 0.8477564102564102, 0.8557692307692308, 0.8545673076923077, 0.8565705128205129, 0.8557692307692308, 0.8569711538461539, 0.8561698717948718, 0.858173076923077, 0.8541666666666666, 0.8505059285013818, 0.8914482481947046, 0.9087991441561916, 0.9087322813586519, 0.9101364001069805, 0.91140679326023, 0.9100249621110814, 0.9079745029865384, 0.9130337880003566, 0.7332572298325724, 0.8116438356164384, 0.8105022831050228, 0.8105022831050228, 0.8116438356164384, 0.8116438356164384, 0.8116438356164384, 0.810882800608828, 0.8116438356164384, 0.625, 0.625, 0.625, 0.5, 0.49924242424242427, 0.625, 0.625, 0.5, 0.625, 0.4995625546806649, 0.5336832895888014, 0.5336832895888014, 0.5336832895888014, 0.5336832895888014, 0.5336832895888014, 0.5336832895888014, 0.5336832895888014, 0.5336832895888014, 0.8745636556088554, 0.890429700695899, 0.8905254149494403, 0.8905479359502736, 0.8904015494448573, 0.8902945746908992, 0.8904353309461073, 0.8904747426975654, 0.8908857509627728, null, null, null, null, null, null, null, null, null, 0.6473913817663818, 0.5302706552706552, 0.6477029914529915, 0.6474358974358975, 0.6477029914529915, 0.6476584757834758, 0.6476139601139601, 0.6476139601139601, 0.6475694444444444, 0.6142857142857142, 0.6158730158730159, 0.6158730158730159, 0.6158730158730159, 0.6158730158730159, 0.6158730158730159, 0.6158730158730159, 0.6158730158730159, 0.6158730158730159, 0.8960620709593702, 0.90427141253118, 0.9055488694506264, 0.9022543752899492, 0.9020325011934137, 0.9020593950232968, 0.9017770098095245, 0.9017635628945829, 0.901440836935986, 0.7759493670886076, 0.7310126582278481, 0.7599683544303797, 0.7748417721518986, 0.7669303797468354, 0.7536392405063291, 0.7805379746835444, 0.7530063291139241, 0.7843354430379748, null, null, null, null, null, null, null, null, null, 0.8650958994708995, 0.8607473544973545, 0.861276455026455, 0.855026455026455, 0.8590939153439153, 0.8545304232804233, 0.8515542328042328, 0.8521825396825397, 0.8537037037037037, 0.9585019535626917, 0.9690174301219474, 0.9712964304140514, 0.9725588230805541, 0.9728648495530487, 0.9724704689808835, 0.9730533382990123, 0.9727529343601329, 0.9727872050412173, 0.9027459283626267, 0.8837559435282396, 0.8852505256679829, 0.8761987794245858, 0.8716930538562417, 0.8702351036316881, 0.8731583304638333, 0.8741620449400336, 0.8705354853362444, 0.956725972264263, 0.9644404645791993, 0.9659567371831522, 0.9674912274801286, 0.9680237446608145, 0.9670021524904988, 0.9665593224139285, 0.9672053498357603, 0.9670161661005168, 0.8633968822986258, 0.8687842865678127, 0.871333713540818, 0.8700063820542068, 0.8735131171582206, 0.877366112650046, 0.8750577100646353, 0.8744330834827005, 0.8773287708435175, 0.9367874793624029, 0.9612229346865653, 0.9586126670936048, 0.9622443437446802, 0.9587356144802297, 0.9631630714689106, 0.9584397301102203, 0.9621281517089687, 0.9580181962132204, 0.4909024745269287, 0.49102377486656973, 0.5066108685104319, 0.4893053533883229, 0.48979055474688676, 0.49029091864790564, 0.4902555393821769, 0.4905335193271874, 0.4906649280284652, 0.7937500000000001, 0.7674479166666667, 0.7385416666666668, 0.7341145833333333, 0.7351562500000001, 0.7341145833333333, 0.7341145833333333, 0.7325520833333333, 0.7346354166666667, 0.8687513763488219, 0.8683109447258313, 0.8683109447258313, 0.8665492182338692, 0.8683109447258314, 0.8694120237833076, 0.8665492182338692, 0.8698524554062981, 0.8685311605373266, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3425219941348973, 0.3425219941348973, 0.3425219941348973, 0.3425219941348973, 0.3425219941348973, 0.3425219941348973, 0.3425219941348973, 0.3425219941348973, 0.3425219941348973, 0.9343141366135496, 0.9469435319790016, 0.9574237800142888, 0.9612444863790264, 0.9628150917901408, 0.9643041577361539, 0.9653204889261641, 0.9656320861678004, 0.9668454710651384, 0.6811936383013608, 0.6811608460403344, 0.6811936383013608, 0.681210034431874, 0.6812264305623872, 0.6811116576487948, 0.6811772421708476, 0.6812100344318741, 0.6811936383013608, 0.8571428571428571, 0.8872448979591837, 0.8821428571428571, 0.8729591836734695, 0.8709183673469387, 0.878061224489796, 0.8755102040816327, 0.8698979591836735, 0.8770408163265306, 0.808962961452588, 0.8207853742086106, 0.8206974420667367, 0.8162116284522923, 0.8123273216634211, 0.8029236799983688, 0.8035137174141324, 0.7996765626433675, 0.7985997125001274], "y0": " ", "yaxis": "y"}],                        {"boxmode": "group", "height": 500, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Distribution of test_roc_auc by number of estimators."}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": " "}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "test_roc_auc"}}},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('c81c2c27-9cc2-46ca-9d7f-54d3b45d6f4a');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

<a id='reference21'></a>
#### Frequency of best hyper-parameter values


```python
plt.figure(figsize=(8,5))

best_param_freq = sns.countplot(best_n_estimators_values['param_value'], palette='Greens_r')

for p in best_param_freq.patches:
    best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

plt.ylabel('frequency')
plt.xlabel('n_estimators')
plt.title('Count of best number of estimators', loc='left')
```




    Text(0.0, 1.0, 'Count of best number of estimators')




![png](output_280_1.png)



```python
# plt.figure(figsize=(12,5))

# best_param_freq = sns.countplot(best_n_estimators_values[best_n_estimators_values.param_value.isin([100,500,1000,2000])]['param_value'], palette='Greens',
#                                 hue=best_n_estimators_values[best_n_estimators_values.param_value.isin([100,500,1000,2000])]['quartile_n_orders'])

# for p in best_param_freq.patches:
#     best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                              ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

# plt.ylabel('frequency')
# plt.xlabel('n_estimators')
# # plt.legend(loc='upper center', title='quartile_n_orders')
# plt.legend(loc='upper left', title='quartile_n_orders', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
# plt.title('Count of best number of estimators', loc='left')
```


```python
plt.figure(figsize=(12,5))

best_param_freq = sns.countplot(best_n_estimators_values[best_n_estimators_values.param_value.isin([100,500,1000,2000])]['quartile_n_orders'], palette='Greens',
                                hue=best_n_estimators_values[best_n_estimators_values.param_value.isin([100,500,1000,2000])]['param_value'])

for p in best_param_freq.patches:
    best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

plt.ylabel('frequency')
plt.xlabel('quartile_n_orders')
# plt.legend(loc='upper center', title='quartile_n_orders')
plt.legend(loc='upper left', title='n_estimators', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
plt.title('Count of best number of estimators', loc='left')
```




    Text(0.0, 1.0, 'Count of best number of estimators')




![png](output_282_1.png)



```python
# plt.figure(figsize=(12,5))

# best_param_freq = sns.countplot(best_n_estimators_values[best_n_estimators_values.param_value.isin([100,500,1000,2000])]['param_value'], palette='Greens',
#                                 hue=best_n_estimators_values[best_n_estimators_values.param_value.isin([100,500,1000,2000])]['quartile_n_vars'])

# for p in best_param_freq.patches:
#     best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                              ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

# plt.ylabel('frequency')
# plt.xlabel('n_estimators')
# plt.legend(loc='upper left', title='quartile_n_vars', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
# plt.title('Count of best number of estimators', loc='left')
```


```python
plt.figure(figsize=(12,5))

best_param_freq = sns.countplot(best_n_estimators_values[best_n_estimators_values.param_value.isin([100,500,1000,2000])]['quartile_n_vars'], palette='Greens',
                                hue=best_n_estimators_values[best_n_estimators_values.param_value.isin([100,500,1000,2000])]['param_value'])

for p in best_param_freq.patches:
    best_param_freq.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'center', xytext = (0, 6), textcoords = 'offset points')

plt.ylabel('frequency')
plt.xlabel('quartile_n_vars')
plt.legend(loc='upper left', title='n_estimators', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
plt.title('Count of best number of estimators', loc='left')
```




    Text(0.0, 1.0, 'Count of best number of estimators')




![png](output_284_1.png)


[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

#### Performance metric against dataset information

When exploring number of estimators, the plot of performance metrics against numbers of observations and features produce the same patters as those obtained when exploring [subsample](#performance_data_info)<a href='#performance_data_info'></a>.

<a id='reference22'></a>
#### Distribution of performance metric by best hyper-parameter value


```python
px.strip(data_frame=best_n_estimators_values.sort_values('param_value'),
         x=best_n_estimators_values.sort_values('param_value')['param_value'].apply(lambda x: 'M = ' + str(x)),
         y=best_n_estimators_values.sort_values('param_value')['test_roc_auc'],
         hover_data=['store_id', 'param_value'],
         color_discrete_sequence=['#0b6fab'],
         width=900, height=500, title='Distribution of test_roc_auc by best number of estimators',
         labels={'y': 'test_roc_auc', 'x': ''})
```


<div>                            <div id="17ce1af9-591c-4f77-a73c-5def6ee908e2" class="plotly-graph-div" style="height:500px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("17ce1af9-591c-4f77-a73c-5def6ee908e2")) {                    Plotly.newPlot(                        "17ce1af9-591c-4f77-a73c-5def6ee908e2",                        [{"alignmentgroup": "True", "boxpoints": "all", "customdata": [[6078.0, 100.0], [4636.0, 100.0], [4601.0, 100.0], [5168.0, 100.0], [4030.0, 100.0], [12658.0, 100.0], [6971.0, 100.0], [7185.0, 100.0], [7292.0, 100.0], [7755.0, 100.0], [5085.0, 100.0], [7939.0, 100.0], [7790.0, 100.0], [8181.0, 100.0], [11729.0, 100.0], [1210.0, 100.0], [8783.0, 100.0], [10650.0, 100.0], [9281.0, 100.0], [10268.0, 100.0], [11223.0, 250.0], [5342.0, 250.0], [5848.0, 250.0], [6966.0, 250.0], [7988.0, 250.0], [10060.0, 250.0], [8832.0, 250.0], [7630.0, 250.0], [8421.0, 250.0], [7845.0, 250.0], [7849.0, 250.0], [6929.0, 250.0], [4974.0, 250.0], [12980.0, 250.0], [3962.0, 250.0], [4408.0, 250.0], [2699.0, 250.0], [1961.0, 250.0], [1739.0, 250.0], [1875.0, 250.0], [7161.0, 500.0], [6970.0, 500.0], [2212.0, 500.0], [1979.0, 500.0], [6714.0, 500.0], [6170.0, 500.0], [2866.0, 500.0], [720.0, 500.0], [9491.0, 500.0], [5663.0, 500.0], [7333.0, 500.0], [1603.0, 750.0], [6106.0, 750.0], [5847.0, 750.0], [8358.0, 750.0], [5593.0, 750.0], [9541.0, 750.0], [1181.0, 750.0], [11723.0, 750.0], [6047.0, 750.0], [8446.0, 1000.0], [1424.0, 1000.0], [4838.0, 1000.0], [4736.0, 1000.0], [2782.0, 1000.0], [3481.0, 1000.0], [5939.0, 1000.0], [3859.0, 1000.0], [6083.0, 1000.0], [1559.0, 1250.0], [3988.0, 1250.0], [9409.0, 1250.0], [8894.0, 1250.0], [10349.0, 1250.0], [4268.0, 1250.0], [6105.0, 1500.0], [10311.0, 1500.0], [5428.0, 1500.0], [2352.0, 1500.0], [8436.0, 1500.0], [1098.0, 1500.0], [3781.0, 1500.0], [2868.0, 1500.0], [6004.0, 1500.0], [6044.0, 1500.0], [9177.0, 1500.0], [6256.0, 1750.0], [9761.0, 1750.0], [3437.0, 1750.0], [5394.0, 1750.0], [1241.0, 1750.0], [5215.0, 2000.0], [3146.0, 2000.0], [8282.0, 2000.0], [2056.0, 2000.0], [8790.0, 2000.0], [9098.0, 2000.0]], "fillcolor": "rgba(255,255,255,0)", "hoveron": "points", "hovertemplate": "=%{x}<br>test_roc_auc=%{y}<br>store_id=%{customdata[0]}<br>param_value=%{customdata[1]}<extra></extra>", "legendgroup": "", "line": {"color": "rgba(255,255,255,0)"}, "marker": {"color": "#0b6fab"}, "name": "", "offsetgroup": "", "orientation": "v", "pointpos": 0, "showlegend": false, "type": "box", "x": ["M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 100.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 250.0", "M = 500.0", "M = 500.0", "M = 500.0", "M = 500.0", "M = 500.0", "M = 500.0", "M = 500.0", "M = 500.0", "M = 500.0", "M = 500.0", "M = 500.0", "M = 750.0", "M = 750.0", "M = 750.0", "M = 750.0", "M = 750.0", "M = 750.0", "M = 750.0", "M = 750.0", "M = 750.0", "M = 1000.0", "M = 1000.0", "M = 1000.0", "M = 1000.0", "M = 1000.0", "M = 1000.0", "M = 1000.0", "M = 1000.0", "M = 1000.0", "M = 1250.0", "M = 1250.0", "M = 1250.0", "M = 1250.0", "M = 1250.0", "M = 1250.0", "M = 1500.0", "M = 1500.0", "M = 1500.0", "M = 1500.0", "M = 1500.0", "M = 1500.0", "M = 1500.0", "M = 1500.0", "M = 1500.0", "M = 1500.0", "M = 1500.0", "M = 1750.0", "M = 1750.0", "M = 1750.0", "M = 1750.0", "M = 1750.0", "M = 2000.0", "M = 2000.0", "M = 2000.0", "M = 2000.0", "M = 2000.0", "M = 2000.0"], "x0": " ", "xaxis": "x", "y": [0.4822222222222222, 0.625, 0.9729511238337575, 0.5777972027972027, 0.9027459283626267, 0.7937500000000001, 0.3425219941348973, 0.7711291098387874, 0.49410377358490565, 0.8650958994708995, 0.7354761904761905, 0.9351963141025641, 0.7034952606635071, 0.48984657039711194, 0.8014500755550389, 0.5, 0.8973077855930147, 0.8625341255821424, 0.88374545220387, 0.5379704301075269, 0.7163408741489128, 0.9374739404320536, 0.9730545876887341, 0.8207853742086106, 0.5783976124885215, 0.8771828115667053, 0.9244421147221805, 0.6158730158730159, 0.5336832895888014, 0.7294730392156864, 0.880878887450629, 0.9276946334089193, 0.8872448979591837, 0.9070022883295195, 0.9676813626931563, 0.76484375, 0.9744435647048474, 0.9206495098039216, 0.8116438356164384, 0.9595803990610329, 0.8137610976594027, 0.9055488694506264, 0.9346271613832853, 0.5066108685104319, 0.6477029914529915, 0.9777860292705839, 0.9448112511215628, 0.8409875306167891, 0.9506790192357779, 0.9543037948167823, 0.9552836190869656, 0.9741010011127673, 0.7142857142857143, 0.9737910860963475, 0.906270215686584, 0.9398820167237061, 0.9563965782391388, 0.982147270672975, 0.9193155452436195, 0.9292351449788291, 0.49713193116634796, 0.9577523647098747, 0.9743088827006509, 0.8961368653421633, 0.6812264305623872, 0.7869261477045908, 0.9943592057761733, 0.9680237446608145, 0.6296079520653943, 0.933179012345679, 0.9877684407096172, 0.9631630714689106, 0.9055077154250708, 0.7994388061722857, 0.877366112650046, 0.9730533382990123, 0.8817414314516129, 0.9141784623327454, 0.9226362703005413, 0.7013870365430781, 0.9713448613803338, 0.9719071441896762, 0.9799159021406727, 0.8920113295628516, 0.9738321651430433, 0.9404545323283953, 0.9425644237453388, 0.6984283677833214, 0.9518518518518518, 0.8698524554062981, 0.858173076923077, 0.8908857509627728, 0.7843354430379748, 0.921678695535953, 0.9668454710651384, 0.8210693760262725, 0.9130337880003566], "y0": " ", "yaxis": "y"}],                        {"boxmode": "group", "height": 500, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Distribution of test_roc_auc by best number of estimators"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": ""}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "test_roc_auc"}}},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('17ce1af9-591c-4f77-a73c-5def6ee908e2');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


[(Main conclusions)](#main_conclusions)<a href='#main_conclusions'></a>

#### Correlation between performance metric of best hyper-parameter value and dataset information


```python
# Generate a mask for the upper triangle:
mask = np.triu(np.ones_like(best_n_estimators_values[['test_roc_auc', 'n_orders', 'n_vars', 'avg_y']].corr(),
                            dtype=np.bool))

sns.heatmap(best_n_estimators_values[['test_roc_auc', 'n_orders', 'n_vars', 'avg_y']].corr(),
            mask = mask, annot = True, cmap = 'viridis')
plt.title('Correlation between performance metric and dataset information')
plt.tight_layout()
```


![png](output_292_0.png)


<a id='grid_search'></a>

## Appendix: grid search with ROC-AUC and average precision score

Definition of hyper-parameters $\theta^L$ for statistical learning methods is usually based on the estimation of errors or performance metrics through validation procedures, after which the best hyper-parameter vector $\theta^{L*}$ among a grid of different alternatives is selected from optimizing the validated error or performance metric.
<br>
When this methodology is referenced by a performance metric instead of error measures, a choice among different performance metrics should be previously done. When the statistical learning problem is a classification one, the choice is especially blurred, since there are several different possibilities. Considering the simplest context of a binary classification, ROC-AUC, precision-recall AUC, and average precision score are the main performance metrics to be considered.
<br>
Given the high correlation between precision-recall AUC and average precision score, and then defining the last one as a representative of both, which of the following should be picked: *ROC-AUC or average precision score*?
<br>
<br>
After estimating a set of models, for a given collection of datasets and one for each learning method explored so far (logistic regression and GBM), their outcomes will be used here to provide comparisons between the best hyper-parameter choice and the associated value of performance metric for those two alternatives, namely, ROC-AUC and average precision score.
<br>
The objectives of this study are summarized as follows:
1. The first objective is to assess changes in the output of grid search as one uses ROC-AUC instead of average precision score as the performance metric of reference. This can be done either by measuring the frequency of divergences or by calculating statistics for the difference in best hyper-parameter values.
    * [Logistic regression](#comparing_param_lr)<a href='#comparing_param_lr'></a>.
    * [GBM](#comparing_param_gbm)<a href='#comparing_param_gbm'></a>.
<br>
<br>
2. When ROC-AUC and average precision score do diverge, even more important is to check the degree to which vary along the alternatives each performance metric itself. This indicates if the divergence in best hyper-parameters choice between the two criteria is meaningful, or instead if both options are closely related. The smaller the expected difference, the more interchangeable are the alternatives.
    * [Logistic regression](#comparing_metric_lr)<a href='#comparing_metric_lr'></a>.
    * [GBM](#comparing_metric_gbm)<a href='#comparing_metric_gbm'></a>.
<br>
<br>
3. Lastly, a last goal is to assess how much the estimated generalization capacity of the models varies between those two options of best hyper-parameters choice, using for this the K-folds CV for implementing grid search and later a train-test estimation in order to calculate performance metrics for final comparison. The results from these tests, however, will be presented and discussed in another notebook.

It is crucial to clarify that there is no desired answers for the questions that arise from these experiments. This because, even if ROC-AUC and average precision score diverge, the definition of which metric is the most appropriate depends ultimately on theoretical discussions to be explored in the future, even though this is also briefly tested here. Note also that the tests whose results are presented below have limitations toward their generalization. From beyond specificities in data generation processes, only two learning methods are explored, and only one hyper-parameter per method is defined through grid search, while GBM, for instance, has at least three further relevant hyper-parameters.

-----------------

#### Main conclusions

**Note:** comparing best hyper-parameters from a learning method $L$ from a dataset $s$ means checking $\hat{\theta}_{roc}^{L,s}$ against $\hat{\theta}_{avg-prec}^{L,s}$, where the first is chosen from grid search based on ROC-AUC and the second follows from grid search using average precision score.
<br>
**Note:** similarly, comparing performance metrics opposes ROC-AUC$_{roc}$ to ROC-AUC$_{avg-prec}$, and $Avg\_prec\_score_{roc}$ to $Avg\_prec\_score_{avg-prec}$, where subscripts $roc$ and $avg-prec$ indicate metrics of performance associated with best hyper-parameters that follow ROC-AUC and average precision score, respectively.
1. Correlation among performance metrics (either using outcomes from logistic regression or GBM).
    * There are high correlations between ROC-AUC and different performance metrics, both precision scores (average precision score and precision-recall AUC - positive correlation) and cost functions (binomial deviance and Brier score - negative correlation). The correlation is particularly high with average precision score.
    * Correlation near 1 between average precision score and precision-recall AUC.
<br>
<br>
2. [Logistic regression](#comp_lr)<a href='#comp_lr'></a>: grid search based on ROC-AUC only moderately diverges in choice of hyper-parameters when comparing to average precision score.
    * Comparing best hyper-parameters:
        * More than a half (55%) of datasets have the same hyper-parameter being chosen across both performance metrics.
        * Given those dataset with divergence in choice, half shows a difference near to 0.25.
        * On average, regularization parameter chosen by ROC-AUC was similar to that picked by average precision score among datasets with divergence in choice, and both was similar to average regularization parameter of datasets with equality in the choice.
        * Distribution of differences (for datasets with divergent choices) concentrates near zero, with only a few of datasets showing large difference in best hyper-parameter.
    * Comparing performance metrics:
        * Average difference in ROC-AUC of aroung 2 percentage points, but with a median of difference smaller than 1 p.p. Again, distribution of differences concentrating around zero, with only a few datasets having large differences.
        * Indeed, high differences in ROC-AUC tipically applying for datasets with ROC-AUC too smaller than those for datasets with equivalent best hyper-parameters.
        * The performance metric of average precision score has a pattern of larger differences.
<br>
<br>
3. [GBM](#comp_gbm)<a href='#comp_gbm'></a>: grid search through ROC-AUC shows results quite divergent than those from grid search through average precision score.
    * Comparing best hyper-parameters:
        * Around a half of datasets (52%) show equal best hyper-parameters between two criteria for grid search.
        * Given datasets with divergent choices, there is a considerable difference in $max\_depth$: on average, a difference of 3.7 splits.
        * ROC-AUC tends to choose the production of larger trees (around 3.6 splits against 2.9 from average precision score).
        * Datasets with no divergence in choice between two criteria are prone to produce smaller trees (around 1.8 splits on average).
        * Highly sparse distribution of differences in $max\_depth$ between that chosen by ROC-AUC and that following average precision score.
    * Comparing performance metrics:
        * The difference in ROC-AUC for best hyper-parameter chosen by ROC-AUC and ROC-AUC for best hyper-parameter chosen by average precision score is not as big as what is suggested by the difference between the best hyper-parameters - 2.6 p.p. of average difference and a median of 1 p.p.
        * The same applies for differences between distinct average precision scores.

<a id='comp_lr'></a>

### Logistic regression

The outcomes from logistic regression model estimation follows from grid-search to define the regularization parameter $\lambda$ (denoted $C$ in sklearn library). The grid explored through train-test estimations has 14 different values for $\lambda$: $\{0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.25, 0.3, 0.5, 0.75, 1.0, 3.0, 10.0\}$. The estimations were performed using 100 different datasets whose response variable was binary.

#### Importing and processing data


```python
# Outcomes from logistic regression estimations:
with open('../Datasets/tun_C.json') as json_file:
    tun_C = json.load(json_file)
```


```python
# Dataframe for correlating different performance metrics:
stores = []
test_roc_auc = []
test_prec_avg = []
test_pr_auc = []
test_deviance_neg = []
test_brier_score_neg = []

# Loop over datasets:
for s in tun_C.keys():
    stores.append(int(s))
    
    # Best hyper-parameter (according with test ROC-AUC):
    roc_auc = list(tun_C[s]['test_roc_auc'].values())
    best_key = list(tun_C[s]['test_roc_auc'].keys())[roc_auc.index(max(roc_auc))]

    # Performance metrics associated with best hyper-parameter:
    test_roc_auc.append(tun_C[s]['test_roc_auc'][best_key])
    test_prec_avg.append(tun_C[s]['test_prec_avg'][best_key])
    test_pr_auc.append(tun_C[s]['test_pr_auc'][best_key])
    test_deviance_neg.append(-tun_C[s]['test_deviance'][best_key])
    test_brier_score_neg.append(-tun_C[s]['test_brier_score'][best_key])

# Dataframe with performance metrics by dataset:
metrics_LR = pd.DataFrame(data={
    'store_id': stores,
    'test_roc_auc': test_roc_auc,
    'test_prec_avg': test_prec_avg,
    'test_pr_auc': test_pr_auc,
    'test_deviance_neg': test_deviance_neg,
    'test_brier_score_neg': test_brier_score_neg
})

print('\033[1mShape of metrics_LR:\033[0m ' + str(metrics_LR.shape) + '.')
metrics_LR.head()
```

    [1mShape of metrics_LR:[0m (100, 6).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>test_roc_auc</th>
      <th>test_prec_avg</th>
      <th>test_pr_auc</th>
      <th>test_deviance_neg</th>
      <th>test_brier_score_neg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11729</td>
      <td>0.860717</td>
      <td>0.261281</td>
      <td>0.255241</td>
      <td>-883.688819</td>
      <td>-0.036629</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10311</td>
      <td>0.759262</td>
      <td>0.056166</td>
      <td>0.042485</td>
      <td>-692.790592</td>
      <td>-0.011738</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7988</td>
      <td>0.675689</td>
      <td>0.163174</td>
      <td>0.155672</td>
      <td>-368.546870</td>
      <td>-0.072928</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4736</td>
      <td>0.992715</td>
      <td>0.840471</td>
      <td>0.834750</td>
      <td>-320.272547</td>
      <td>-0.016298</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3481</td>
      <td>0.789703</td>
      <td>0.089113</td>
      <td>0.076277</td>
      <td>-1515.658151</td>
      <td>-0.009110</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dataframe with best hyper-parameters and associated performance metrics:
stores = []
best_param_roc_auc = []
best_param_avg_prec = []
roc_auc_best_roc_auc = []
roc_auc_best_avg_prec = []
avg_prec_best_avg_prec = []
avg_prec_best_roc_auc = []

# Loop over datasets:
for s in tun_C.keys():
    stores.append(int(s))
    
    # Hyper-parameters and performance metrics:
    params = list(tun_C[s]['test_roc_auc'].keys())
    test_roc_auc = list(tun_C[s]['test_roc_auc'].values())
    test_prec_avg = list(tun_C[s]['test_prec_avg'].values())

    # Best hyper-parameters:
    best_param_roc_auc.append(float(params[test_roc_auc.index(max(test_roc_auc))]))
    best_param_avg_prec.append(float(params[test_prec_avg.index(max(test_prec_avg))]))
    
    # Performance metrics for best hyper-parameters:
    roc_auc_best_roc_auc.append(float(test_roc_auc[test_roc_auc.index(max(test_roc_auc))]))
    roc_auc_best_avg_prec.append(float(test_roc_auc[test_prec_avg.index(max(test_prec_avg))]))
    avg_prec_best_avg_prec.append(float(test_prec_avg[test_prec_avg.index(max(test_prec_avg))]))
    avg_prec_best_roc_auc.append(float(test_prec_avg[test_roc_auc.index(max(test_roc_auc))]))

# Dataframe with information of best hyper-parameters and associated performance metrics by dataset:
grid_search_comp_LR = pd.DataFrame(data={
    'store_id': stores,
    'best_param_roc_auc': best_param_roc_auc,
    'best_param_avg_prec': best_param_avg_prec,
    'roc_auc_best_roc_auc': roc_auc_best_roc_auc,
    'roc_auc_best_avg_prec': roc_auc_best_avg_prec,
    'avg_prec_best_avg_prec': avg_prec_best_avg_prec,
    'avg_prec_best_roc_auc': avg_prec_best_roc_auc
})

print('\033[1mShape of grid_search_comp_LR:\033[0m ' + str(grid_search_comp_LR.shape) + '.')
grid_search_comp_LR.head()
```

    [1mShape of grid_search_comp_LR:[0m (100, 7).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>best_param_roc_auc</th>
      <th>best_param_avg_prec</th>
      <th>roc_auc_best_roc_auc</th>
      <th>roc_auc_best_avg_prec</th>
      <th>avg_prec_best_avg_prec</th>
      <th>avg_prec_best_roc_auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11729</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.860717</td>
      <td>0.860717</td>
      <td>0.261281</td>
      <td>0.261281</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10311</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.759262</td>
      <td>0.759262</td>
      <td>0.056166</td>
      <td>0.056166</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7988</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.675689</td>
      <td>0.675689</td>
      <td>0.163174</td>
      <td>0.163174</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4736</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.992715</td>
      <td>0.991722</td>
      <td>0.850960</td>
      <td>0.840471</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3481</td>
      <td>0.3</td>
      <td>10.0</td>
      <td>0.789703</td>
      <td>0.743769</td>
      <td>0.209854</td>
      <td>0.089113</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dataframe with datasets for which best hyper-parameters differ between ROC-AUC and average precision score:
diff_param_LR = grid_search_comp_LR[grid_search_comp_LR.best_param_roc_auc != grid_search_comp_LR.best_param_avg_prec]
diff_param_LR['abs_diff_best_param'] = np.abs(diff_param_LR.best_param_roc_auc - diff_param_LR.best_param_avg_prec)
diff_param_LR['abs_diff_roc_auc'] = np.abs(diff_param_LR.roc_auc_best_roc_auc - diff_param_LR.roc_auc_best_avg_prec)
diff_param_LR['abs_diff_avg_prec'] = np.abs(diff_param_LR.avg_prec_best_avg_prec - diff_param_LR.avg_prec_best_roc_auc)

print('\033[1mShape of diff_param_LR:\033[0m ' + str(diff_param_LR.shape) + '.')
diff_param_LR.sort_values('abs_diff_best_param', ascending=False).head(10)
```

    C:\Users\Acer\Miniconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    
    C:\Users\Acer\Miniconda3\lib\site-packages\ipykernel_launcher.py:4: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    
    

    [1mShape of diff_param_LR:[0m (44, 10).
    

    C:\Users\Acer\Miniconda3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>best_param_roc_auc</th>
      <th>best_param_avg_prec</th>
      <th>roc_auc_best_roc_auc</th>
      <th>roc_auc_best_avg_prec</th>
      <th>avg_prec_best_avg_prec</th>
      <th>avg_prec_best_roc_auc</th>
      <th>abs_diff_best_param</th>
      <th>abs_diff_roc_auc</th>
      <th>abs_diff_avg_prec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>71</th>
      <td>8446</td>
      <td>0.0001</td>
      <td>10.00</td>
      <td>0.500000</td>
      <td>0.432122</td>
      <td>0.001681</td>
      <td>0.000955</td>
      <td>9.9999</td>
      <td>0.067878</td>
      <td>0.000726</td>
    </tr>
    <tr>
      <th>83</th>
      <td>3146</td>
      <td>10.0000</td>
      <td>0.10</td>
      <td>0.823101</td>
      <td>0.719620</td>
      <td>0.259247</td>
      <td>0.092862</td>
      <td>9.9000</td>
      <td>0.103481</td>
      <td>0.166385</td>
    </tr>
    <tr>
      <th>91</th>
      <td>1979</td>
      <td>0.3000</td>
      <td>10.00</td>
      <td>0.813036</td>
      <td>0.791748</td>
      <td>0.048136</td>
      <td>0.021521</td>
      <td>9.7000</td>
      <td>0.021288</td>
      <td>0.026615</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3481</td>
      <td>0.3000</td>
      <td>10.00</td>
      <td>0.789703</td>
      <td>0.743769</td>
      <td>0.209854</td>
      <td>0.089113</td>
      <td>9.7000</td>
      <td>0.045934</td>
      <td>0.120741</td>
    </tr>
    <tr>
      <th>55</th>
      <td>5428</td>
      <td>10.0000</td>
      <td>0.75</td>
      <td>0.963344</td>
      <td>0.959472</td>
      <td>0.619015</td>
      <td>0.591090</td>
      <td>9.2500</td>
      <td>0.003872</td>
      <td>0.027925</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10268</td>
      <td>1.0000</td>
      <td>10.00</td>
      <td>0.694892</td>
      <td>0.610215</td>
      <td>0.118871</td>
      <td>0.090646</td>
      <td>9.0000</td>
      <td>0.084677</td>
      <td>0.028225</td>
    </tr>
    <tr>
      <th>57</th>
      <td>5939</td>
      <td>10.0000</td>
      <td>3.00</td>
      <td>0.971570</td>
      <td>0.963899</td>
      <td>0.466383</td>
      <td>0.433333</td>
      <td>7.0000</td>
      <td>0.007671</td>
      <td>0.033049</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2782</td>
      <td>0.3000</td>
      <td>3.00</td>
      <td>0.863650</td>
      <td>0.831612</td>
      <td>0.245908</td>
      <td>0.189285</td>
      <td>2.7000</td>
      <td>0.032038</td>
      <td>0.056623</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4736</td>
      <td>1.0000</td>
      <td>3.00</td>
      <td>0.992715</td>
      <td>0.991722</td>
      <td>0.850960</td>
      <td>0.840471</td>
      <td>2.0000</td>
      <td>0.000993</td>
      <td>0.010489</td>
    </tr>
    <tr>
      <th>35</th>
      <td>7849</td>
      <td>1.0000</td>
      <td>3.00</td>
      <td>0.892595</td>
      <td>0.889210</td>
      <td>0.384869</td>
      <td>0.368977</td>
      <td>2.0000</td>
      <td>0.003385</td>
      <td>0.015892</td>
    </tr>
  </tbody>
</table>
</div>



#### Correlation between performance metrics


```python
# Generate a mask for the upper triangle:
mask = np.triu(np.ones_like(metrics_LR.drop('store_id', axis=1).corr(), dtype=np.bool))

plt.figure(figsize=(8,6))
sns.heatmap(metrics_LR.drop('store_id', axis=1).corr(), mask = mask, annot = True, cmap = 'viridis')
plt.title('Correlation between performance metrics', loc='left')
plt.tight_layout()
```


![png](output_309_0.png)


<a id='comparing_param_lr'></a>

#### Comparing best hyper-parameters


```python
print('\033[1mNumber of datasets with no difference in best hyper-parameter:\033[0m ' +
      str(len(grid_search_comp_LR.dropna()) - len(diff_param_LR)) + ' (' +
      str(round(((len(grid_search_comp_LR.dropna()) - len(diff_param_LR))/len(grid_search_comp_LR.dropna()))*100,2)) +
      '%).')
```

    [1mNumber of datasets with no difference in best hyper-parameter:[0m 53 (54.64%).
    


```python
print('\033[1mComparing best hyper-parameter:\033[0m')
print('\n')
print('\033[1mStatistics for absolute difference of best hyper-parameter:\033[0m')
print(diff_param_LR.abs_diff_best_param.describe())

diff_param_LR.sort_values('abs_diff_best_param', ascending=False).head(10)
```

    [1mComparing best hyper-parameter:[0m
    
    
    [1mStatistics for absolute difference of best hyper-parameter:[0m
    count    44.000000
    mean      1.893111
    std       3.299265
    min       0.027000
    25%       0.150000
    50%       0.260000
    75%       1.175000
    max       9.999900
    Name: abs_diff_best_param, dtype: float64
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>best_param_roc_auc</th>
      <th>best_param_avg_prec</th>
      <th>roc_auc_best_roc_auc</th>
      <th>roc_auc_best_avg_prec</th>
      <th>avg_prec_best_avg_prec</th>
      <th>avg_prec_best_roc_auc</th>
      <th>abs_diff_best_param</th>
      <th>abs_diff_roc_auc</th>
      <th>abs_diff_avg_prec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>71</th>
      <td>8446</td>
      <td>0.0001</td>
      <td>10.00</td>
      <td>0.500000</td>
      <td>0.432122</td>
      <td>0.001681</td>
      <td>0.000955</td>
      <td>9.9999</td>
      <td>0.067878</td>
      <td>0.000726</td>
    </tr>
    <tr>
      <th>83</th>
      <td>3146</td>
      <td>10.0000</td>
      <td>0.10</td>
      <td>0.823101</td>
      <td>0.719620</td>
      <td>0.259247</td>
      <td>0.092862</td>
      <td>9.9000</td>
      <td>0.103481</td>
      <td>0.166385</td>
    </tr>
    <tr>
      <th>91</th>
      <td>1979</td>
      <td>0.3000</td>
      <td>10.00</td>
      <td>0.813036</td>
      <td>0.791748</td>
      <td>0.048136</td>
      <td>0.021521</td>
      <td>9.7000</td>
      <td>0.021288</td>
      <td>0.026615</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3481</td>
      <td>0.3000</td>
      <td>10.00</td>
      <td>0.789703</td>
      <td>0.743769</td>
      <td>0.209854</td>
      <td>0.089113</td>
      <td>9.7000</td>
      <td>0.045934</td>
      <td>0.120741</td>
    </tr>
    <tr>
      <th>55</th>
      <td>5428</td>
      <td>10.0000</td>
      <td>0.75</td>
      <td>0.963344</td>
      <td>0.959472</td>
      <td>0.619015</td>
      <td>0.591090</td>
      <td>9.2500</td>
      <td>0.003872</td>
      <td>0.027925</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10268</td>
      <td>1.0000</td>
      <td>10.00</td>
      <td>0.694892</td>
      <td>0.610215</td>
      <td>0.118871</td>
      <td>0.090646</td>
      <td>9.0000</td>
      <td>0.084677</td>
      <td>0.028225</td>
    </tr>
    <tr>
      <th>57</th>
      <td>5939</td>
      <td>10.0000</td>
      <td>3.00</td>
      <td>0.971570</td>
      <td>0.963899</td>
      <td>0.466383</td>
      <td>0.433333</td>
      <td>7.0000</td>
      <td>0.007671</td>
      <td>0.033049</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2782</td>
      <td>0.3000</td>
      <td>3.00</td>
      <td>0.863650</td>
      <td>0.831612</td>
      <td>0.245908</td>
      <td>0.189285</td>
      <td>2.7000</td>
      <td>0.032038</td>
      <td>0.056623</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4736</td>
      <td>1.0000</td>
      <td>3.00</td>
      <td>0.992715</td>
      <td>0.991722</td>
      <td>0.850960</td>
      <td>0.840471</td>
      <td>2.0000</td>
      <td>0.000993</td>
      <td>0.010489</td>
    </tr>
    <tr>
      <th>35</th>
      <td>7849</td>
      <td>1.0000</td>
      <td>3.00</td>
      <td>0.892595</td>
      <td>0.889210</td>
      <td>0.384869</td>
      <td>0.368977</td>
      <td>2.0000</td>
      <td>0.003385</td>
      <td>0.015892</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mStatistics of best hyper-parameter for datasets with difference in best hyper-parameter:\033[0m')
print('\033[1mBest hyper-parameter according with test ROC-AUC:\033[0m')
grid_search_comp_LR[grid_search_comp_LR.best_param_roc_auc != grid_search_comp_LR.best_param_avg_prec][['best_param_roc_auc']].describe().transpose()
```

    [1mStatistics of best hyper-parameter for datasets with difference in best hyper-parameter:[0m
    [1mBest hyper-parameter according with test ROC-AUC:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>best_param_roc_auc</th>
      <td>44.0</td>
      <td>1.033707</td>
      <td>2.468934</td>
      <td>0.0001</td>
      <td>0.25</td>
      <td>0.3</td>
      <td>0.5625</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mStatistics of best hyper-parameter for datasets with difference in best hyper-parameter:\033[0m')
print('\033[1mBest hyper-parameter according with test average precision score:\033[0m')
grid_search_comp_LR[grid_search_comp_LR.best_param_roc_auc != grid_search_comp_LR.best_param_avg_prec][['best_param_avg_prec']].describe().transpose()
```

    [1mStatistics of best hyper-parameter for datasets with difference in best hyper-parameter:[0m
    [1mBest hyper-parameter according with test average precision score:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>best_param_avg_prec</th>
      <td>44.0</td>
      <td>1.585909</td>
      <td>2.830271</td>
      <td>0.03</td>
      <td>0.2125</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mStatistics of best hyper-parameter for datasets with no difference in best hyper-parameter:\033[0m')
grid_search_comp_LR[grid_search_comp_LR.best_param_roc_auc == grid_search_comp_LR.best_param_avg_prec][['best_param_roc_auc']].describe().transpose()
```

    [1mStatistics of best hyper-parameter for datasets with no difference in best hyper-parameter:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>best_param_roc_auc</th>
      <td>56.0</td>
      <td>1.130005</td>
      <td>2.830953</td>
      <td>0.0001</td>
      <td>0.1</td>
      <td>0.25</td>
      <td>0.3</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Distribution of absolute differences in best hyper-parameter:
plt.figure(figsize=(7,5))

diff_param_LR.abs_diff_best_param.hist(bins=50)

plt.title('Distribution of abs_diff_best_param', loc='left')
```




    Text(0.0, 1.0, 'Distribution of abs_diff_best_param')




![png](output_317_1.png)


<a id='comparing_metric_lr'></a>

#### Comparing performance metrics


```python
print('\033[1mComparing test ROC-AUC:\033[0m')
print('\n')
print('\033[1mStatistics for absolute difference of ROC-AUC:\033[0m')
print(diff_param_LR.abs_diff_roc_auc.describe())

diff_param_LR.sort_values('abs_diff_roc_auc', ascending=False).head(10)
```

    [1mComparing test ROC-AUC:[0m
    
    
    [1mStatistics for absolute difference of ROC-AUC:[0m
    count    44.000000
    mean      0.018372
    std       0.027125
    min       0.000016
    25%       0.001934
    50%       0.005848
    75%       0.022499
    max       0.103481
    Name: abs_diff_roc_auc, dtype: float64
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>best_param_roc_auc</th>
      <th>best_param_avg_prec</th>
      <th>roc_auc_best_roc_auc</th>
      <th>roc_auc_best_avg_prec</th>
      <th>avg_prec_best_avg_prec</th>
      <th>avg_prec_best_roc_auc</th>
      <th>abs_diff_best_param</th>
      <th>abs_diff_roc_auc</th>
      <th>abs_diff_avg_prec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>83</th>
      <td>3146</td>
      <td>10.0000</td>
      <td>0.10</td>
      <td>0.823101</td>
      <td>0.719620</td>
      <td>0.259247</td>
      <td>0.092862</td>
      <td>9.9000</td>
      <td>0.103481</td>
      <td>0.166385</td>
    </tr>
    <tr>
      <th>29</th>
      <td>7845</td>
      <td>0.5000</td>
      <td>0.10</td>
      <td>0.833742</td>
      <td>0.746426</td>
      <td>0.239007</td>
      <td>0.175623</td>
      <td>0.4000</td>
      <td>0.087316</td>
      <td>0.063384</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10268</td>
      <td>1.0000</td>
      <td>10.00</td>
      <td>0.694892</td>
      <td>0.610215</td>
      <td>0.118871</td>
      <td>0.090646</td>
      <td>9.0000</td>
      <td>0.084677</td>
      <td>0.028225</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5848</td>
      <td>0.1000</td>
      <td>1.00</td>
      <td>0.934959</td>
      <td>0.857375</td>
      <td>0.326670</td>
      <td>0.251868</td>
      <td>0.9000</td>
      <td>0.077584</td>
      <td>0.074802</td>
    </tr>
    <tr>
      <th>71</th>
      <td>8446</td>
      <td>0.0001</td>
      <td>10.00</td>
      <td>0.500000</td>
      <td>0.432122</td>
      <td>0.001681</td>
      <td>0.000955</td>
      <td>9.9999</td>
      <td>0.067878</td>
      <td>0.000726</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4408</td>
      <td>0.2500</td>
      <td>0.75</td>
      <td>0.805990</td>
      <td>0.759635</td>
      <td>0.276317</td>
      <td>0.269647</td>
      <td>0.5000</td>
      <td>0.046354</td>
      <td>0.006670</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3481</td>
      <td>0.3000</td>
      <td>10.00</td>
      <td>0.789703</td>
      <td>0.743769</td>
      <td>0.209854</td>
      <td>0.089113</td>
      <td>9.7000</td>
      <td>0.045934</td>
      <td>0.120741</td>
    </tr>
    <tr>
      <th>49</th>
      <td>7790</td>
      <td>0.1000</td>
      <td>0.25</td>
      <td>0.612757</td>
      <td>0.572966</td>
      <td>0.400270</td>
      <td>0.370552</td>
      <td>0.1500</td>
      <td>0.039791</td>
      <td>0.029718</td>
    </tr>
    <tr>
      <th>72</th>
      <td>8790</td>
      <td>0.3000</td>
      <td>1.00</td>
      <td>0.825226</td>
      <td>0.791769</td>
      <td>0.405025</td>
      <td>0.397106</td>
      <td>0.7000</td>
      <td>0.033456</td>
      <td>0.007920</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2782</td>
      <td>0.3000</td>
      <td>3.00</td>
      <td>0.863650</td>
      <td>0.831612</td>
      <td>0.245908</td>
      <td>0.189285</td>
      <td>2.7000</td>
      <td>0.032038</td>
      <td>0.056623</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Distribution of absolute differences in test ROC-AUC:
plt.figure(figsize=(7,5))

diff_param_LR.abs_diff_roc_auc.hist(bins=50)

plt.title('Distribution of abs_diff_roc_auc', loc='left')
```




    Text(0.0, 1.0, 'Distribution of abs_diff_roc_auc')




![png](output_321_1.png)



```python
print('\033[1mStatistics of test ROC-AUC for datasets with the largest absolute difference:\033[0m')
diff_param_LR[diff_param_LR.abs_diff_roc_auc > diff_param_LR.abs_diff_roc_auc.quantile(q=0.75)][['roc_auc_best_roc_auc']].describe().transpose()
```

    [1mStatistics of test ROC-AUC for datasets with the largest absolute difference:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>roc_auc_best_roc_auc</th>
      <td>11.0</td>
      <td>0.778425</td>
      <td>0.127236</td>
      <td>0.5</td>
      <td>0.742298</td>
      <td>0.823101</td>
      <td>0.848696</td>
      <td>0.934959</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mStatistics of test ROC-AUC for datasets with difference in best hyper-parameter:\033[0m')
grid_search_comp_LR[grid_search_comp_LR.best_param_roc_auc != grid_search_comp_LR.best_param_avg_prec][['roc_auc_best_roc_auc']].describe().transpose()
```

    [1mStatistics of test ROC-AUC for datasets with difference in best hyper-parameter:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>roc_auc_best_roc_auc</th>
      <td>44.0</td>
      <td>0.873927</td>
      <td>0.125698</td>
      <td>0.5</td>
      <td>0.831613</td>
      <td>0.919596</td>
      <td>0.962979</td>
      <td>0.993563</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mStatistics of test ROC-AUC for datasets with no difference in best hyper-parameter:\033[0m')
grid_search_comp_LR[grid_search_comp_LR.best_param_roc_auc == grid_search_comp_LR.best_param_avg_prec][['roc_auc_best_roc_auc']].describe().transpose()
```

    [1mStatistics of test ROC-AUC for datasets with no difference in best hyper-parameter:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>roc_auc_best_roc_auc</th>
      <td>53.0</td>
      <td>0.909007</td>
      <td>0.08542</td>
      <td>0.625</td>
      <td>0.89837</td>
      <td>0.930476</td>
      <td>0.970886</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mComparing test average precision score:\033[0m')
print('\n')
print('\033[1mStatistics for absolute difference of average precision score:\033[0m')
print(diff_param_LR.abs_diff_avg_prec.describe())

diff_param_LR.sort_values('abs_diff_avg_prec', ascending=False).head(10)
```

    [1mComparing test average precision score:[0m
    
    
    [1mStatistics for absolute difference of average precision score:[0m
    count    44.000000
    mean      0.028780
    std       0.034506
    min       0.000287
    25%       0.004544
    50%       0.018033
    75%       0.035037
    max       0.166385
    Name: abs_diff_avg_prec, dtype: float64
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>best_param_roc_auc</th>
      <th>best_param_avg_prec</th>
      <th>roc_auc_best_roc_auc</th>
      <th>roc_auc_best_avg_prec</th>
      <th>avg_prec_best_avg_prec</th>
      <th>avg_prec_best_roc_auc</th>
      <th>abs_diff_best_param</th>
      <th>abs_diff_roc_auc</th>
      <th>abs_diff_avg_prec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>83</th>
      <td>3146</td>
      <td>10.00</td>
      <td>0.1</td>
      <td>0.823101</td>
      <td>0.719620</td>
      <td>0.259247</td>
      <td>0.092862</td>
      <td>9.90</td>
      <td>0.103481</td>
      <td>0.166385</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3481</td>
      <td>0.30</td>
      <td>10.0</td>
      <td>0.789703</td>
      <td>0.743769</td>
      <td>0.209854</td>
      <td>0.089113</td>
      <td>9.70</td>
      <td>0.045934</td>
      <td>0.120741</td>
    </tr>
    <tr>
      <th>81</th>
      <td>7630</td>
      <td>0.10</td>
      <td>0.3</td>
      <td>0.669048</td>
      <td>0.661111</td>
      <td>0.303037</td>
      <td>0.204032</td>
      <td>0.20</td>
      <td>0.007937</td>
      <td>0.099005</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5848</td>
      <td>0.10</td>
      <td>1.0</td>
      <td>0.934959</td>
      <td>0.857375</td>
      <td>0.326670</td>
      <td>0.251868</td>
      <td>0.90</td>
      <td>0.077584</td>
      <td>0.074802</td>
    </tr>
    <tr>
      <th>60</th>
      <td>10650</td>
      <td>0.30</td>
      <td>1.0</td>
      <td>0.885392</td>
      <td>0.879316</td>
      <td>0.362141</td>
      <td>0.295101</td>
      <td>0.70</td>
      <td>0.006076</td>
      <td>0.067039</td>
    </tr>
    <tr>
      <th>29</th>
      <td>7845</td>
      <td>0.50</td>
      <td>0.1</td>
      <td>0.833742</td>
      <td>0.746426</td>
      <td>0.239007</td>
      <td>0.175623</td>
      <td>0.40</td>
      <td>0.087316</td>
      <td>0.063384</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2782</td>
      <td>0.30</td>
      <td>3.0</td>
      <td>0.863650</td>
      <td>0.831612</td>
      <td>0.245908</td>
      <td>0.189285</td>
      <td>2.70</td>
      <td>0.032038</td>
      <td>0.056623</td>
    </tr>
    <tr>
      <th>26</th>
      <td>6047</td>
      <td>0.25</td>
      <td>0.1</td>
      <td>0.948414</td>
      <td>0.948377</td>
      <td>0.434277</td>
      <td>0.383996</td>
      <td>0.15</td>
      <td>0.000037</td>
      <td>0.050281</td>
    </tr>
    <tr>
      <th>21</th>
      <td>10060</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>0.872246</td>
      <td>0.870384</td>
      <td>0.612258</td>
      <td>0.563001</td>
      <td>0.75</td>
      <td>0.001862</td>
      <td>0.049258</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4838</td>
      <td>0.03</td>
      <td>0.3</td>
      <td>0.972613</td>
      <td>0.966992</td>
      <td>0.638555</td>
      <td>0.596136</td>
      <td>0.27</td>
      <td>0.005620</td>
      <td>0.042418</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Distribution of absolute differences in test average precision score:
plt.figure(figsize=(7,5))

diff_param_LR.abs_diff_avg_prec.hist(bins=50)

plt.title('Distribution of abs_diff_avg_prec', loc='left')
```




    Text(0.0, 1.0, 'Distribution of abs_diff_avg_prec')




![png](output_326_1.png)



```python
print('\033[1mStatistics of test average precision score for datasets with the largest absolute difference:\033[0m')
diff_param_LR[diff_param_LR.abs_diff_avg_prec > diff_param_LR.abs_diff_avg_prec.quantile(q=0.75)][['avg_prec_best_avg_prec']].describe().transpose()
```

    [1mStatistics of test average precision score for datasets with the largest absolute difference:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>avg_prec_best_avg_prec</th>
      <td>11.0</td>
      <td>0.391447</td>
      <td>0.17314</td>
      <td>0.209854</td>
      <td>0.252578</td>
      <td>0.32667</td>
      <td>0.523268</td>
      <td>0.674959</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mStatistics of test average precision score for datasets with difference in best hyper-parameter:\033[0m')
grid_search_comp_LR[grid_search_comp_LR.best_param_roc_auc != grid_search_comp_LR.best_param_avg_prec][['avg_prec_best_avg_prec']].describe().transpose()
```

    [1mStatistics of test average precision score for datasets with difference in best hyper-parameter:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>avg_prec_best_avg_prec</th>
      <td>44.0</td>
      <td>0.459934</td>
      <td>0.247255</td>
      <td>0.001681</td>
      <td>0.27205</td>
      <td>0.45033</td>
      <td>0.684618</td>
      <td>0.913769</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mStatistics of test average precision score for datasets with no difference in best hyper-parameter:\033[0m')
grid_search_comp_LR[grid_search_comp_LR.best_param_roc_auc == grid_search_comp_LR.best_param_avg_prec][['avg_prec_best_avg_prec']].describe().transpose()
```

    [1mStatistics of test average precision score for datasets with no difference in best hyper-parameter:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>avg_prec_best_avg_prec</th>
      <td>53.0</td>
      <td>0.582064</td>
      <td>0.27004</td>
      <td>0.005235</td>
      <td>0.40405</td>
      <td>0.608948</td>
      <td>0.79521</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



<a id='comp_gbm'></a>

### GBM

The outcomes from GBM estimation follows from grid-search to define the regularization parameter $\max\_depth$, the maximum number of splits in the threes composing the ensemble. The grid explored through train-test estimations has 10 different values for $\max\_depth$: $\{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$. The estimations were performed using 100 different datasets whose response variable was binary. The remaining most relevant hyper-parameters were defined as follows: $\eta = 1$, $v = 0.1$, and $n\_estimators = 500$.

#### Importing and processing data


```python
# Outcomes from GBM estimations:
with open('../Datasets/tun_subsample.json') as json_file:
    tun_subsample = json.load(json_file)
```


```python
# Dataframe with performance metrics by dataset:
metrics_GBM = pd.DataFrame(data={
    'store_id': [int(s) for s in tun_subsample.keys() if np.isnan(tun_subsample[s]['test_roc_auc']['0.75']) == False],
    'test_roc_auc': [tun_subsample[s]['test_roc_auc']['0.75'] for s in tun_subsample.keys() if np.isnan(tun_subsample[s]['test_roc_auc']['0.75']) == False],
    'test_prec_avg': [tun_subsample[s]['test_prec_avg']['0.75'] for s in tun_subsample.keys() if np.isnan(tun_subsample[s]['test_prec_avg']['0.75']) == False],
    'test_pr_auc': [tun_subsample[s]['test_pr_auc']['0.75'] for s in tun_subsample.keys() if np.isnan(tun_subsample[s]['test_pr_auc']['0.75']) == False],
    'test_deviance_neg': [-tun_subsample[s]['test_deviance']['0.75'] for s in tun_subsample.keys() if np.isnan(tun_subsample[s]['test_deviance']['0.75']) == False],
    'test_brier_score_neg': [-tun_subsample[s]['test_brier_score']['0.75'] for s in tun_subsample.keys() if np.isnan(tun_subsample[s]['test_brier_score']['0.75']) == False]
})

print('\033[1mShape of metrics_GBM:\033[0m ' + str(metrics_GBM.shape) + '.')
metrics_GBM.head()
```

    [1mShape of metrics_GBM:[0m (97, 6).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>test_roc_auc</th>
      <th>test_prec_avg</th>
      <th>test_pr_auc</th>
      <th>test_deviance_neg</th>
      <th>test_brier_score_neg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11729</td>
      <td>0.801276</td>
      <td>0.191084</td>
      <td>0.184379</td>
      <td>-885.512767</td>
      <td>-0.043436</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10311</td>
      <td>0.799521</td>
      <td>0.023465</td>
      <td>0.019710</td>
      <td>-693.145870</td>
      <td>-0.008808</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7988</td>
      <td>0.563636</td>
      <td>0.091626</td>
      <td>0.087912</td>
      <td>-372.231960</td>
      <td>-0.096923</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4736</td>
      <td>0.921247</td>
      <td>0.628839</td>
      <td>0.624813</td>
      <td>-322.896852</td>
      <td>-0.027266</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3481</td>
      <td>0.729311</td>
      <td>0.039222</td>
      <td>0.032138</td>
      <td>-1517.080351</td>
      <td>-0.008447</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dataframe with best hyper-parameters and associated performance metrics:
stores = []
best_param_roc_auc = []
best_param_avg_prec = []
roc_auc_best_roc_auc = []
roc_auc_best_avg_prec = []
avg_prec_best_avg_prec = []
avg_prec_best_roc_auc = []

# Loop over datasets:
for s in tun_max_depth['500'].keys():
    stores.append(int(s))
    
    # Hyper-parameters and performance metrics:
    params = list(tun_max_depth['500'][s]['test_roc_auc'].keys())
    test_roc_auc = list(tun_max_depth['500'][s]['test_roc_auc'].values())
    test_prec_avg = list(tun_max_depth['500'][s]['test_prec_avg'].values())

    # Best hyper-parameters:
    best_param_roc_auc.append(float(params[test_roc_auc.index(max(test_roc_auc))]))
    best_param_avg_prec.append(float(params[test_prec_avg.index(max(test_prec_avg))]))
    
    # Performance metrics for best hyper-parameters:
    roc_auc_best_roc_auc.append(float(test_roc_auc[test_roc_auc.index(max(test_roc_auc))]))
    roc_auc_best_avg_prec.append(float(test_roc_auc[test_prec_avg.index(max(test_prec_avg))]))
    avg_prec_best_avg_prec.append(float(test_prec_avg[test_prec_avg.index(max(test_prec_avg))]))
    avg_prec_best_roc_auc.append(float(test_prec_avg[test_roc_auc.index(max(test_roc_auc))]))

# Dataframe with information of best hyper-parameters and associated performance metrics by dataset:
grid_search_comp_GBM = pd.DataFrame(data={
    'store_id': stores,
    'best_param_roc_auc': best_param_roc_auc,
    'best_param_avg_prec': best_param_avg_prec,
    'roc_auc_best_roc_auc': roc_auc_best_roc_auc,
    'roc_auc_best_avg_prec': roc_auc_best_avg_prec,
    'avg_prec_best_avg_prec': avg_prec_best_avg_prec,
    'avg_prec_best_roc_auc': avg_prec_best_roc_auc
})

print('\033[1mShape of grid_search_comp_GBM:\033[0m ' + str(grid_search_comp_GBM.shape) + '.')
grid_search_comp_GBM.head()
```

    [1mShape of grid_search_comp_GBM:[0m (100, 7).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>best_param_roc_auc</th>
      <th>best_param_avg_prec</th>
      <th>roc_auc_best_roc_auc</th>
      <th>roc_auc_best_avg_prec</th>
      <th>avg_prec_best_avg_prec</th>
      <th>avg_prec_best_roc_auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11729</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.799801</td>
      <td>0.799801</td>
      <td>0.202104</td>
      <td>0.202104</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10311</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>0.887601</td>
      <td>0.766948</td>
      <td>0.052554</td>
      <td>0.038844</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7988</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>0.583999</td>
      <td>0.583999</td>
      <td>0.123488</td>
      <td>0.123488</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4736</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>0.908057</td>
      <td>0.900055</td>
      <td>0.562723</td>
      <td>0.456224</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3481</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.781987</td>
      <td>0.781987</td>
      <td>0.110081</td>
      <td>0.110081</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dataframe with datasets for which best hyper-parameters differ between ROC-AUC and average precision score:
diff_param_GBM = grid_search_comp_GBM[grid_search_comp_GBM.best_param_roc_auc != grid_search_comp_GBM.best_param_avg_prec]
diff_param_GBM['abs_diff_best_param'] = np.abs(diff_param_GBM.best_param_roc_auc - diff_param_GBM.best_param_avg_prec)
diff_param_GBM['abs_diff_roc_auc'] = np.abs(diff_param_GBM.roc_auc_best_roc_auc - diff_param_GBM.roc_auc_best_avg_prec)
diff_param_GBM['abs_diff_avg_prec'] = np.abs(diff_param_GBM.avg_prec_best_avg_prec - diff_param_GBM.avg_prec_best_roc_auc)

print('\033[1mShape of diff_param_GBM:\033[0m ' + str(diff_param_GBM.shape) + '.')
diff_param_GBM.sort_values('abs_diff_best_param', ascending=False).head(10)
```

    C:\Users\Acer\Miniconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    
    C:\Users\Acer\Miniconda3\lib\site-packages\ipykernel_launcher.py:4: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    
    C:\Users\Acer\Miniconda3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    
    

    [1mShape of diff_param_GBM:[0m (47, 10).
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>best_param_roc_auc</th>
      <th>best_param_avg_prec</th>
      <th>roc_auc_best_roc_auc</th>
      <th>roc_auc_best_avg_prec</th>
      <th>avg_prec_best_avg_prec</th>
      <th>avg_prec_best_roc_auc</th>
      <th>abs_diff_best_param</th>
      <th>abs_diff_roc_auc</th>
      <th>abs_diff_avg_prec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>85</th>
      <td>7755</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>0.863327</td>
      <td>0.841733</td>
      <td>0.532300</td>
      <td>0.502225</td>
      <td>9.0</td>
      <td>0.021594</td>
      <td>0.030074</td>
    </tr>
    <tr>
      <th>35</th>
      <td>7849</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>0.893840</td>
      <td>0.887102</td>
      <td>0.400260</td>
      <td>0.341014</td>
      <td>8.0</td>
      <td>0.006738</td>
      <td>0.059246</td>
    </tr>
    <tr>
      <th>93</th>
      <td>5394</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>0.872055</td>
      <td>0.868861</td>
      <td>0.627453</td>
      <td>0.545424</td>
      <td>8.0</td>
      <td>0.003193</td>
      <td>0.082029</td>
    </tr>
    <tr>
      <th>27</th>
      <td>8832</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>0.927887</td>
      <td>0.910926</td>
      <td>0.378259</td>
      <td>0.317216</td>
      <td>8.0</td>
      <td>0.016961</td>
      <td>0.061043</td>
    </tr>
    <tr>
      <th>70</th>
      <td>11723</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>0.923893</td>
      <td>0.909745</td>
      <td>0.447972</td>
      <td>0.361462</td>
      <td>7.0</td>
      <td>0.014148</td>
      <td>0.086511</td>
    </tr>
    <tr>
      <th>99</th>
      <td>6966</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>0.824191</td>
      <td>0.802755</td>
      <td>0.478225</td>
      <td>0.475037</td>
      <td>5.0</td>
      <td>0.021435</td>
      <td>0.003188</td>
    </tr>
    <tr>
      <th>87</th>
      <td>4030</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.922828</td>
      <td>0.904266</td>
      <td>0.380409</td>
      <td>0.223817</td>
      <td>5.0</td>
      <td>0.018561</td>
      <td>0.156592</td>
    </tr>
    <tr>
      <th>53</th>
      <td>2212</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.917246</td>
      <td>0.859159</td>
      <td>0.458835</td>
      <td>0.341768</td>
      <td>5.0</td>
      <td>0.058087</td>
      <td>0.117067</td>
    </tr>
    <tr>
      <th>83</th>
      <td>3146</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>0.880854</td>
      <td>0.767880</td>
      <td>0.191062</td>
      <td>0.146340</td>
      <td>5.0</td>
      <td>0.112975</td>
      <td>0.044723</td>
    </tr>
    <tr>
      <th>13</th>
      <td>6044</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>0.966470</td>
      <td>0.960556</td>
      <td>0.542730</td>
      <td>0.524035</td>
      <td>5.0</td>
      <td>0.005914</td>
      <td>0.018694</td>
    </tr>
  </tbody>
</table>
</div>



#### Correlation between performance metrics


```python
# Generate a mask for the upper triangle:
mask = np.triu(np.ones_like(metrics_GBM.drop('store_id', axis=1).corr(), dtype=np.bool))

plt.figure(figsize=(8,6))
sns.heatmap(metrics_GBM.drop('store_id', axis=1).corr(), mask = mask, annot = True, cmap = 'viridis')
plt.title('Correlation between performance metrics', loc='left')
plt.tight_layout()
```


![png](output_339_0.png)


<a id='comparing_param_gbm'></a>

#### Comparing best hyper-parameters


```python
print('\033[1mNumber of datasets with no difference in best hyper-parameter:\033[0m ' +
      str(len(grid_search_comp_GBM.dropna()) - len(diff_param_GBM)) + ' (' +
      str(round(((len(grid_search_comp_GBM.dropna()) - len(diff_param_GBM))/len(grid_search_comp_GBM.dropna()))*100,2)) +
      '%).')
```

    [1mNumber of datasets with no difference in best hyper-parameter:[0m 50 (51.55%).
    


```python
print('\033[1mComparing best hyper-parameter:\033[0m')
print('\n')
print('\033[1mStatistics for absolute difference of best hyper-parameter:\033[0m')
print(diff_param_GBM.abs_diff_best_param.describe())

diff_param_GBM.sort_values('abs_diff_best_param', ascending=False).head(10)
```

    [1mComparing best hyper-parameter:[0m
    
    
    [1mStatistics for absolute difference of best hyper-parameter:[0m
    count    47.000000
    mean      3.148936
    std       2.176742
    min       1.000000
    25%       1.000000
    50%       3.000000
    75%       4.000000
    max       9.000000
    Name: abs_diff_best_param, dtype: float64
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>best_param_roc_auc</th>
      <th>best_param_avg_prec</th>
      <th>roc_auc_best_roc_auc</th>
      <th>roc_auc_best_avg_prec</th>
      <th>avg_prec_best_avg_prec</th>
      <th>avg_prec_best_roc_auc</th>
      <th>abs_diff_best_param</th>
      <th>abs_diff_roc_auc</th>
      <th>abs_diff_avg_prec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>85</th>
      <td>7755</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>0.863327</td>
      <td>0.841733</td>
      <td>0.532300</td>
      <td>0.502225</td>
      <td>9.0</td>
      <td>0.021594</td>
      <td>0.030074</td>
    </tr>
    <tr>
      <th>35</th>
      <td>7849</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>0.893840</td>
      <td>0.887102</td>
      <td>0.400260</td>
      <td>0.341014</td>
      <td>8.0</td>
      <td>0.006738</td>
      <td>0.059246</td>
    </tr>
    <tr>
      <th>93</th>
      <td>5394</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>0.872055</td>
      <td>0.868861</td>
      <td>0.627453</td>
      <td>0.545424</td>
      <td>8.0</td>
      <td>0.003193</td>
      <td>0.082029</td>
    </tr>
    <tr>
      <th>27</th>
      <td>8832</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>0.927887</td>
      <td>0.910926</td>
      <td>0.378259</td>
      <td>0.317216</td>
      <td>8.0</td>
      <td>0.016961</td>
      <td>0.061043</td>
    </tr>
    <tr>
      <th>70</th>
      <td>11723</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>0.923893</td>
      <td>0.909745</td>
      <td>0.447972</td>
      <td>0.361462</td>
      <td>7.0</td>
      <td>0.014148</td>
      <td>0.086511</td>
    </tr>
    <tr>
      <th>99</th>
      <td>6966</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>0.824191</td>
      <td>0.802755</td>
      <td>0.478225</td>
      <td>0.475037</td>
      <td>5.0</td>
      <td>0.021435</td>
      <td>0.003188</td>
    </tr>
    <tr>
      <th>87</th>
      <td>4030</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.922828</td>
      <td>0.904266</td>
      <td>0.380409</td>
      <td>0.223817</td>
      <td>5.0</td>
      <td>0.018561</td>
      <td>0.156592</td>
    </tr>
    <tr>
      <th>53</th>
      <td>2212</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.917246</td>
      <td>0.859159</td>
      <td>0.458835</td>
      <td>0.341768</td>
      <td>5.0</td>
      <td>0.058087</td>
      <td>0.117067</td>
    </tr>
    <tr>
      <th>83</th>
      <td>3146</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>0.880854</td>
      <td>0.767880</td>
      <td>0.191062</td>
      <td>0.146340</td>
      <td>5.0</td>
      <td>0.112975</td>
      <td>0.044723</td>
    </tr>
    <tr>
      <th>13</th>
      <td>6044</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>0.966470</td>
      <td>0.960556</td>
      <td>0.542730</td>
      <td>0.524035</td>
      <td>5.0</td>
      <td>0.005914</td>
      <td>0.018694</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mStatistics of best hyper-parameter for datasets with difference in best hyper-parameter:\033[0m')
print('\033[1mBest hyper-parameter according with test ROC-AUC:\033[0m')
grid_search_comp_GBM[grid_search_comp_GBM.best_param_roc_auc != grid_search_comp_GBM.best_param_avg_prec][['best_param_roc_auc']].describe().transpose()
```

    [1mStatistics of best hyper-parameter for datasets with difference in best hyper-parameter:[0m
    [1mBest hyper-parameter according with test ROC-AUC:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>best_param_roc_auc</th>
      <td>47.0</td>
      <td>3.574468</td>
      <td>2.551595</td>
      <td>1.0</td>
      <td>1.5</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mStatistics of best hyper-parameter for datasets with difference in best hyper-parameter:\033[0m')
print('\033[1mBest hyper-parameter according with test average precision score:\033[0m')
grid_search_comp_GBM[grid_search_comp_GBM.best_param_roc_auc != grid_search_comp_GBM.best_param_avg_prec][['best_param_avg_prec']].describe().transpose()
```

    [1mStatistics of best hyper-parameter for datasets with difference in best hyper-parameter:[0m
    [1mBest hyper-parameter according with test average precision score:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>best_param_avg_prec</th>
      <td>47.0</td>
      <td>2.851064</td>
      <td>2.466988</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mStatistics of best hyper-parameter for datasets with no difference in best hyper-parameter:\033[0m')
grid_search_comp_GBM[grid_search_comp_GBM.best_param_roc_auc == grid_search_comp_GBM.best_param_avg_prec][['best_param_roc_auc']].describe().transpose()
```

    [1mStatistics of best hyper-parameter for datasets with no difference in best hyper-parameter:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>best_param_roc_auc</th>
      <td>53.0</td>
      <td>1.792453</td>
      <td>1.70247</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Distribution of absolute differences in best hyper-parameter:
plt.figure(figsize=(7,5))

diff_param_GBM.abs_diff_best_param.hist(bins=50)

plt.title('Distribution of abs_diff_best_param', loc='left')
```




    Text(0.0, 1.0, 'Distribution of abs_diff_best_param')




![png](output_347_1.png)


<a id='comparing_metric_gbm'></a>

#### Comparing performance metrics


```python
print('\033[1mComparing test ROC-AUC:\033[0m')
print('\n')
print('\033[1mStatistics for absolute difference of ROC-AUC:\033[0m')
print(diff_param_GBM.abs_diff_roc_auc.describe())

diff_param_GBM.sort_values('abs_diff_roc_auc', ascending=False).head(10)
```

    [1mComparing test ROC-AUC:[0m
    
    
    [1mStatistics for absolute difference of ROC-AUC:[0m
    count    47.000000
    mean      0.026472
    std       0.040345
    min       0.000277
    25%       0.003975
    50%       0.010529
    75%       0.023492
    max       0.177540
    Name: abs_diff_roc_auc, dtype: float64
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>best_param_roc_auc</th>
      <th>best_param_avg_prec</th>
      <th>roc_auc_best_roc_auc</th>
      <th>roc_auc_best_avg_prec</th>
      <th>avg_prec_best_avg_prec</th>
      <th>avg_prec_best_roc_auc</th>
      <th>abs_diff_best_param</th>
      <th>abs_diff_roc_auc</th>
      <th>abs_diff_avg_prec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65</th>
      <td>7185</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.720124</td>
      <td>0.542584</td>
      <td>0.170330</td>
      <td>0.045479</td>
      <td>1.0</td>
      <td>0.177540</td>
      <td>0.124851</td>
    </tr>
    <tr>
      <th>54</th>
      <td>9761</td>
      <td>4.0</td>
      <td>8.0</td>
      <td>0.738872</td>
      <td>0.610005</td>
      <td>0.039437</td>
      <td>0.034038</td>
      <td>4.0</td>
      <td>0.128867</td>
      <td>0.005399</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10311</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>0.887601</td>
      <td>0.766948</td>
      <td>0.052554</td>
      <td>0.038844</td>
      <td>2.0</td>
      <td>0.120653</td>
      <td>0.013710</td>
    </tr>
    <tr>
      <th>29</th>
      <td>7845</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.848958</td>
      <td>0.729677</td>
      <td>0.202312</td>
      <td>0.174654</td>
      <td>2.0</td>
      <td>0.119281</td>
      <td>0.027658</td>
    </tr>
    <tr>
      <th>83</th>
      <td>3146</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>0.880854</td>
      <td>0.767880</td>
      <td>0.191062</td>
      <td>0.146340</td>
      <td>5.0</td>
      <td>0.112975</td>
      <td>0.044723</td>
    </tr>
    <tr>
      <th>77</th>
      <td>8421</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.500000</td>
      <td>0.425634</td>
      <td>0.004197</td>
      <td>0.003487</td>
      <td>2.0</td>
      <td>0.074366</td>
      <td>0.000709</td>
    </tr>
    <tr>
      <th>53</th>
      <td>2212</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.917246</td>
      <td>0.859159</td>
      <td>0.458835</td>
      <td>0.341768</td>
      <td>5.0</td>
      <td>0.058087</td>
      <td>0.117067</td>
    </tr>
    <tr>
      <th>25</th>
      <td>8783</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>0.894203</td>
      <td>0.850336</td>
      <td>0.499345</td>
      <td>0.499056</td>
      <td>4.0</td>
      <td>0.043868</td>
      <td>0.000288</td>
    </tr>
    <tr>
      <th>68</th>
      <td>3962</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.967088</td>
      <td>0.933852</td>
      <td>0.559490</td>
      <td>0.460256</td>
      <td>4.0</td>
      <td>0.033236</td>
      <td>0.099234</td>
    </tr>
    <tr>
      <th>80</th>
      <td>6714</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.678619</td>
      <td>0.647703</td>
      <td>0.279895</td>
      <td>0.188249</td>
      <td>2.0</td>
      <td>0.030916</td>
      <td>0.091646</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Distribution of absolute differences in test ROC-AUC:
plt.figure(figsize=(7,5))

diff_param_GBM.abs_diff_roc_auc.hist(bins=50)

plt.title('Distribution of abs_diff_roc_auc', loc='left')
```




    Text(0.0, 1.0, 'Distribution of abs_diff_roc_auc')




![png](output_351_1.png)



```python
print('\033[1mStatistics of test ROC-AUC for datasets with the largest absolute difference:\033[0m')
diff_param_GBM[diff_param_GBM.abs_diff_roc_auc > diff_param_GBM.abs_diff_roc_auc.quantile(q=0.75)][['roc_auc_best_roc_auc']].describe().transpose()
```

    [1mStatistics of test ROC-AUC for datasets with the largest absolute difference:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>roc_auc_best_roc_auc</th>
      <td>12.0</td>
      <td>0.781848</td>
      <td>0.14532</td>
      <td>0.5</td>
      <td>0.709748</td>
      <td>0.814844</td>
      <td>0.889251</td>
      <td>0.967088</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mStatistics of test ROC-AUC for datasets with difference in best hyper-parameter:\033[0m')
grid_search_comp_GBM[grid_search_comp_GBM.best_param_roc_auc != grid_search_comp_GBM.best_param_avg_prec][['roc_auc_best_roc_auc']].describe().transpose()
```

    [1mStatistics of test ROC-AUC for datasets with difference in best hyper-parameter:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>roc_auc_best_roc_auc</th>
      <td>47.0</td>
      <td>0.854242</td>
      <td>0.142226</td>
      <td>0.482222</td>
      <td>0.831678</td>
      <td>0.900665</td>
      <td>0.956614</td>
      <td>0.987558</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mStatistics of test ROC-AUC for datasets with no difference in best hyper-parameter:\033[0m')
grid_search_comp_GBM[grid_search_comp_GBM.best_param_roc_auc == grid_search_comp_GBM.best_param_avg_prec][['roc_auc_best_roc_auc']].describe().transpose()
```

    [1mStatistics of test ROC-AUC for datasets with no difference in best hyper-parameter:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>roc_auc_best_roc_auc</th>
      <td>50.0</td>
      <td>0.863824</td>
      <td>0.122886</td>
      <td>0.5</td>
      <td>0.804156</td>
      <td>0.903295</td>
      <td>0.947715</td>
      <td>0.989621</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mComparing test average precision score:\033[0m')
print('\n')
print('\033[1mStatistics for absolute difference of average precision score:\033[0m')
print(diff_param_GBM.abs_diff_avg_prec.describe())

diff_param_GBM.sort_values('abs_diff_avg_prec', ascending=False).head(10)
```

    [1mComparing test average precision score:[0m
    
    
    [1mStatistics for absolute difference of average precision score:[0m
    count    47.000000
    mean      0.048202
    std       0.056130
    min       0.000000
    25%       0.005111
    50%       0.025520
    75%       0.084270
    max       0.199456
    Name: abs_diff_avg_prec, dtype: float64
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>best_param_roc_auc</th>
      <th>best_param_avg_prec</th>
      <th>roc_auc_best_roc_auc</th>
      <th>roc_auc_best_avg_prec</th>
      <th>avg_prec_best_avg_prec</th>
      <th>avg_prec_best_roc_auc</th>
      <th>abs_diff_best_param</th>
      <th>abs_diff_roc_auc</th>
      <th>abs_diff_avg_prec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>4838</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.975498</td>
      <td>0.972562</td>
      <td>0.564125</td>
      <td>0.364669</td>
      <td>4.0</td>
      <td>0.002936</td>
      <td>0.199456</td>
    </tr>
    <tr>
      <th>22</th>
      <td>6256</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.905136</td>
      <td>0.901163</td>
      <td>0.329840</td>
      <td>0.140069</td>
      <td>1.0</td>
      <td>0.003973</td>
      <td>0.189772</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2782</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>0.700844</td>
      <td>0.691613</td>
      <td>0.285545</td>
      <td>0.120405</td>
      <td>3.0</td>
      <td>0.009231</td>
      <td>0.165140</td>
    </tr>
    <tr>
      <th>87</th>
      <td>4030</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.922828</td>
      <td>0.904266</td>
      <td>0.380409</td>
      <td>0.223817</td>
      <td>5.0</td>
      <td>0.018561</td>
      <td>0.156592</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4408</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>0.780729</td>
      <td>0.755339</td>
      <td>0.288852</td>
      <td>0.154125</td>
      <td>1.0</td>
      <td>0.025391</td>
      <td>0.134727</td>
    </tr>
    <tr>
      <th>65</th>
      <td>7185</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.720124</td>
      <td>0.542584</td>
      <td>0.170330</td>
      <td>0.045479</td>
      <td>1.0</td>
      <td>0.177540</td>
      <td>0.124851</td>
    </tr>
    <tr>
      <th>92</th>
      <td>12658</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>0.827865</td>
      <td>0.824479</td>
      <td>0.520173</td>
      <td>0.400740</td>
      <td>2.0</td>
      <td>0.003385</td>
      <td>0.119433</td>
    </tr>
    <tr>
      <th>53</th>
      <td>2212</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.917246</td>
      <td>0.859159</td>
      <td>0.458835</td>
      <td>0.341768</td>
      <td>5.0</td>
      <td>0.058087</td>
      <td>0.117067</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4736</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>0.908057</td>
      <td>0.900055</td>
      <td>0.562723</td>
      <td>0.456224</td>
      <td>4.0</td>
      <td>0.008002</td>
      <td>0.106499</td>
    </tr>
    <tr>
      <th>68</th>
      <td>3962</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.967088</td>
      <td>0.933852</td>
      <td>0.559490</td>
      <td>0.460256</td>
      <td>4.0</td>
      <td>0.033236</td>
      <td>0.099234</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Distribution of absolute differences in test average precision score:
plt.figure(figsize=(7,5))

diff_param_GBM.abs_diff_avg_prec.hist(bins=50)

plt.title('Distribution of abs_diff_avg_prec', loc='left')
```




    Text(0.0, 1.0, 'Distribution of abs_diff_avg_prec')




![png](output_356_1.png)



```python
print('\033[1mStatistics of test average precision score for datasets with the largest absolute difference:\033[0m')
diff_param_GBM[diff_param_GBM.abs_diff_avg_prec > diff_param_GBM.abs_diff_avg_prec.quantile(q=0.75)][['avg_prec_best_avg_prec']].describe().transpose()
```

    [1mStatistics of test average precision score for datasets with the largest absolute difference:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>avg_prec_best_avg_prec</th>
      <td>12.0</td>
      <td>0.404016</td>
      <td>0.133756</td>
      <td>0.17033</td>
      <td>0.288025</td>
      <td>0.414191</td>
      <td>0.530002</td>
      <td>0.564125</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mStatistics of test average precision score for datasets with difference in best hyper-parameter:\033[0m')
grid_search_comp_GBM[grid_search_comp_GBM.best_param_roc_auc != grid_search_comp_GBM.best_param_avg_prec][['avg_prec_best_avg_prec']].describe().transpose()
```

    [1mStatistics of test average precision score for datasets with difference in best hyper-parameter:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>avg_prec_best_avg_prec</th>
      <td>47.0</td>
      <td>0.433623</td>
      <td>0.259853</td>
      <td>0.000957</td>
      <td>0.238833</td>
      <td>0.461855</td>
      <td>0.563424</td>
      <td>0.951548</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('\033[1mStatistics of test average precision score for datasets with no difference in best hyper-parameter:\033[0m')
grid_search_comp_GBM[grid_search_comp_GBM.best_param_roc_auc == grid_search_comp_GBM.best_param_avg_prec][['avg_prec_best_avg_prec']].describe().transpose()
```

    [1mStatistics of test average precision score for datasets with no difference in best hyper-parameter:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>avg_prec_best_avg_prec</th>
      <td>50.0</td>
      <td>0.464446</td>
      <td>0.271804</td>
      <td>0.001986</td>
      <td>0.234535</td>
      <td>0.448691</td>
      <td>0.702354</td>
      <td>0.963264</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
