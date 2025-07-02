# NFISiS: new fuzzy inference systems

## Project description

Author: Kaike Sa Teles Rocha Alves

NFISiS (new fuzzy inference systems) is a package that contains new machine learning models developed by Kaike Alves during his PhD research. 

    Author: Kaike Sa Teles Rocha Alves (PhD)
    Email: kaikerochaalves@outlook.com or kaike.alves@estudante.ufjf.br

Doi for preprint: http://dx.doi.org/10.48550/arXiv.2506.06285

Doi to cite the code: http://dx.doi.org/10.5281/zenodo.15746843

Github repository: https://github.com/kaikerochaalves/NFISiS_PyPi

Doi for the thesis: http://dx.doi.org/10.13140/RG.2.2.25910.00324

It provides:

1. New Mamdani Classifier: NewMamadaniClassifier
2. New Mamdani Regressor: NewMamdaniRegressor
3. New Takagi-Sugeno-Kang: NTSK
4. Genetic NMR: GEN_NMR
5. Genetic NTSK: GEN_NTSK
6. Random NMR: R_NMR
7. Random NTSK: R_NTSK

Cite: SA TELES ROCHA ALVES, K., & Pestana de Aguiar, E. (2025). NFISiS: new fuzzy inference systems. Zenodo. https://doi.org/10.5281/zenodo.15746843

## Description:

NFISiS: A Novel Python Library for Interpretable Time Series Forecasting and Classification

NFISiS (New Fuzzy Inference Systems) is a groundbreaking Python library available on PyPI (https://pypi.org/project/nfisis/) that introduces a suite of advanced fuzzy inference systems. This library is specifically designed to tackle the complexities of time series forecasting and classification problems by offering machine learning models that prioritize both high accuracy and enhanced interpretability/explainability.

At its core, NFISiS features novel data-driven Mamdani and Takagi-Sugeno-Kang (TSK) fuzzy models. These models are further empowered by the integration of cutting-edge techniques, including:

*- Genetic Algorithms: Employed for intelligent feature selection, optimizing model performance, and boosting interpretability by identifying the most relevant attributes in complex datasets.*

*- Ensemble Methods: Utilized to combine multiple fuzzy models, leading to superior predictive accuracy and increased robustness against overfitting.*

Unlike many black-box machine learning approaches, NFISiS stands out by providing clear, understandable insights into its decision-making process. This unique combination of advanced fuzzy logic, genetic algorithms, and ensemble techniques allows NFISiS to achieve superior performance across various challenging datasets, including renewable energy, finance, and cryptocurrency applications.

Choose NFISiS to develop powerful and transparent machine learning solutions for your time series analysis needs.

## Instructions

To install the library use the command: 

    pip install nfisis

The library provides 6 models in fuzzy systems, as follows:

### NewMamdaniClassifier (NMC)

NewMamdaniClassifier is based on Mamdani and applied for classification problems

To import the NewMamdaniClassifier (NMC), simply type the command:

    from nfisis.fuzzy import NewMamdaniClassifier as NMC

Hyperparameters:

rules : int, default=5 - Number of fuzzy rules that will be created.

fuzzy_operator : {'prod', 'max', 'min', 'minmax'}, default='prod'
Choose the fuzzy operator:

*- 'prod' will use :`product`*

*- 'max' will use :class:`maximum value`*

*- 'min' will use :class:`minimum value`*

*- 'minmax' will use :class:`minimum value multiplied by maximum`*

Both hyperparameters are important for performance

Example of NewMamdaniClassifier (NMC):

    from nfisis.fuzzy import NewMamdaniClassifier as NMC
    model = NMC(rules = 4, fuzzy_operator = "min")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

### NewMamdaniRegressor (NMR)

To import the NewMamdaniRegressor (NMR), simply type:

    from nfisis.fuzzy import NewMamdaniRegressor as NMR

Hyperparameters
    
rules : int, default=5
Number of fuzzy rules that will be created.

fuzzy_operator : {'prod', 'max', 'min', 'equal'}, default='prod'
Choose the fuzzy operator:

*- 'prod' will use :`product`*

*- 'max' will use :class:`maximum value`*

*- 'min' will use :class:`minimum value`*

*- 'minmax' will use :class:`minimum value multiplied by maximum`*

*- 'equal' use the same firing degree for all rules*

ponder : boolean, default=True
ponder controls whether the firing degree of each fuzzy rule 
is weighted by the number of observations (data points) 
associated with that rule during the tau calculation.
Used to avoid the influence of less representative rules

Example of NewMamdaniRegressor (NMR):

    from nfisis.fuzzy import NewMamdaniRegressor as NMR
    model = NMR(rules = 4, fuzzy_operator = "max", ponder=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

### New Takagi-Sugeno-Kang (NTSK)

To import the NTSK (New Takagi-Sugeno-Kang), type:

    from nfisis.fuzzy import NTSK

Hyperparameters
    
rules : int, default=5
Number of fuzzy rules will be created.

lambda1 : float, possible values are in the interval [0,1], default=1
Defines the forgetting factor for the algorithm to estimate the consequent parameters.
This parameters is only used when RLS_option is "RLS"

adaptive_filter : {'RLS', 'wRLS'}, default='RLS'
Algorithm used to compute the consequent parameters:

*- 'RLS' will use :class:`RLS`*

*- 'wRLS' will use :class:`wRLS`*

fuzzy_operator : {'prod', 'max', 'min', 'minmax'}, default='prod'
Choose the fuzzy operator:

*- 'prod' will use :`product`*

*- 'max' will use :class:`maximum value`*

*- 'min' will use :class:`minimum value`*

*- 'minmax' will use :class:`minimum value multiplied by maximum`*

omega : int, default=1000
Omega is a parameters used to initialize the algorithm to estimate
the consequent parameters

ponder : boolean, default=True
ponder controls whether the firing degree of each fuzzy rule 
is weighted by the number of observations (data points) 
associated with that rule during the tau calculation.
Used to avoid the influence of less representative rules

NTSK usually have lower errors than NMR because it uses polynomial functions, however it tends to be less explainable.

Notes: 

1. When using adaptive_filter = "RLS", all rules have the same consequent parameters.
2. When using adaptive_filter="wRLS", the consequent parameters of each rule is adjusted differently by a factor weight.
3. Only use lambda1 when you choose adaptive_filter = 'RLS'.
4. omega is not very relevant for performance.

Example of NTSK (RLS):

    from nfisis.fuzzy import NTSK
    model = NTSK(rules = 4, lambda1= 0.99, adaptive_filter = "RLS", fuzzy_operator = "minmax", ponder=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

Example of NTSK (wRLS):

    from nfisis.fuzzy import NTSK
    model = NTSK(rules = 4, adaptive_filter = "wRLS", fuzzy_operator = "prod", ponder=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

NewMandaniClassifier, NewMamdaniRegressor, and NTSK are new data-driven fuzzy models that automatically create fuzzy rules and fuzzy sets. You can learn more about this models in papers: https://doi.org/10.1016/j.engappai.2024.108155, https://doi.org/10.1007/s10614-024-10670-w, http://dx.doi.org/10.48550/arXiv.2506.06285, and http://dx.doi.org/10.13140/RG.2.2.25910.00324

The library nfisis also includes the NTSK and NMR (NewMandaniRegressor) with genetic-algorithm as attribute selection. At this time, the paper containing the proposal of these models are fourthcoming.

### Genetic NMR (GEN-NMR)

To import GEN-NMR type:

    from nfisis.genetic import GEN_NMR

Hyperparameters
    
rules : int, default=5
Number of fuzzy rules that will be created.

fuzzy_operator : {'prod', 'max', 'min', 'equal'}, default='prod'
Choose the fuzzy operator:

*- 'prod' will use :`product`*

*- 'max' will use :class:`maximum value`*

*- 'min' will use :class:`minimum value`*

*- 'minmax' will use :class:`minimum value multiplied by maximum`*

*- 'equal' use the same firing degree for all rules*

ponder : boolean, default=True
If True, the firing degree of each fuzzy rule will be weighted by the number of observations
associated with that rule. This gives more influence to rules derived from a larger
number of training data points. If False, all rules contribute equally regardless
of their observation count.

num_generations : int, default=10
Number of generations the genetic algorithm will run. A higher number of generations
allows the algorithm to explore more solutions and potentially find a better one,
but increases computation time.

num_parents_mating : int, default=5
Number of parents that will be selected to mate in each generation.
These parents are chosen based on their fitness values to produce offspring.

sol_per_pop : int, default=10
Number of solutions (individuals) in the population for the genetic algorithm.
A larger population can increase the diversity of solutions explored,
but also increases computational cost per generation.

error_metric : {'RMSE', 'NRMSE', 'NDEI', 'MAE', 'MAPE'}, default='RMSE'
The error metric used as the fitness function for the genetic algorithm.
The genetic algorithm aims to minimize this metric (by maximizing its negative value).

*- 'RMSE': Root Mean Squared Error.*

*- 'NRMSE': Normalized Root Mean Squared Error.*

*- 'NDEI': Non-Dimensional Error Index.*

*- 'MAE': Mean Absolute Error.*

*- 'MAPE': Mean Absolute Percentage Error.*


print_information : bool, default=False
If True, information about the genetic algorithm's progress (e.g., generation number,
current fitness, and fitness change) will be printed during the `fit` process.

parallel_processing : list or None, default=None
Configuration for parallel processing using PyGAD's capabilities.
Refer to PyGAD's documentation for valid formats. If None, parallel processing
is not used. 

*- parallel_processing=None: no parallel processing is applied,*

*- parallel_processing=['process', 10]: applies parallel processing with 10 processes,*

*- parallel_processing=['thread', 5] or parallel_processing=5: applies parallel processing with 5 threads.*


Example of GEN-NMR:

    from nfisis.genetic import GEN_NMR
    model = GEN_NMR(rules = 3, fuzzy_operator = "minmax", ponder = False, num_generations = 20, num_parents_mating = 10, sol_per_pop = 10, error_metric = "MAE", print_information=True, parallel_processing=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

### Genetic NTSK (GEN-NTSK)

To import GEN-NTSK type:

    from nfisis.genetic import GEN_NTSK

Hyperparameters
    
rules : int, default=5
Number of fuzzy rules will be created.

lambda1 : float, possible values are in the interval [0,1], default=1
Defines the forgetting factor for the algorithm to estimate the consequent parameters.
This parameters is only used when RLS_option is "RLS"

adaptive_filter : {'RLS', 'wRLS'}, default='wRLS'
Algorithm used to compute the consequent parameters:

*- 'RLS' will use :class:`RLS`*

*- 'wRLS' will use :class:`wRLS`*


fuzzy_operator : {'prod', 'max', 'min', 'minmax'}, default='prod'
Choose the fuzzy operator:

*- 'prod' will use :`product`*

*- 'max' will use :class:`maximum value`*

*- 'min' will use :class:`minimum value`*

*- 'minmax' will use :class:`minimum value multiplied by maximum`*

omega : int, default=1000
Omega is a parameters used to initialize the algorithm to estimate
the consequent parameters

ponder : bool, default=True
If True, the firing degree of each fuzzy rule will be weighted by the number of observations
associated with that rule. This gives more influence to rules derived from a larger
number of training data points. If False, all rules contribute equally regardless
of their observation count.

num_generations : int, default=10
Number of generations the genetic algorithm will run. A higher number of generations
allows the algorithm to explore more solutions and potentially find a better one,
but increases computation time.

num_parents_mating : int, default=5
Number of parents that will be selected to mate in each generation.
These parents are chosen based on their fitness values to produce offspring.

sol_per_pop : int, default=10
Number of solutions (individuals) in the population for the genetic algorithm.
A larger population can increase the diversity of solutions explored,
but also increases computational cost per generation.

error_metric : {'RMSE', 'NRMSE', 'NDEI', 'MAE', 'MAPE'}, default='RMSE'
The error metric used as the fitness function for the genetic algorithm.
The genetic algorithm aims to minimize this metric (by maximizing its negative value).

*- 'RMSE': Root Mean Squared Error.*

*- 'NRMSE': Normalized Root Mean Squared Error.*

*- 'NDEI': Non-Dimensional Error Index.*

*- 'MAE': Mean Absolute Error.*

*- 'MAPE': Mean Absolute Percentage Error.*


print_information : bool, default=False
If True, information about the genetic algorithm's progress (e.g., generation number,
current fitness, and fitness change) will be printed during the `fit` process.

parallel_processing : list or None, default=None
Configuration for parallel processing using PyGAD's capabilities.
Refer to PyGAD's documentation for valid formats. If None, parallel processing
is not used. 

*- parallel_processing=None: no parallel processing is applied,*

*- parallel_processing=['process', 10]: applies parallel processing with 10 processes,*

*- parallel_processing=['thread', 5] or parallel_processing=5: applies parallel processing with 5 threads.*


Example of GEN-NTSK:

    from nfisis.genetic import GEN_NTSK
    model = GEN_NTSK(rules = 6, error_metric = "MAE", print_information=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

Finally, one last inovation of this library that was part of the reasearch of the PhD of Kaike Alves and it is in his forthcoming thesis is the ensemble model with fuzzy systems, reffered as to R_NMR and R_NTSK:

### Random NMR (R-NMR)

    from nfisis.ensemble import R_NMR

Hyperparameters:

n_estimators : int, default=100
The number of individual models (estimators) that will be generated
and combined to form the ensemble. A higher number of estimators
generally leads to a more robust and accurate ensemble but increases
training time. Think of this as how many "experts" you're gathering
to make a final decision.

n_trials : int, default=5
For each estimator in the ensemble, this parameter specifies the
number of attempts (trials) to find the best-performing underlying
model and its optimal feature subset. More trials increase the
chances of discovering a better individual model, but it also means
more computational effort.

combination : {'mean', 'median', 'weighted_average'}, default='mean'
This hyperparameter dictates the technique used to combine the
predictions from all the individual estimators in the ensemble into
a single final prediction.

*- 'mean': The final prediction is the simple average of all individual
model predictions. This is a straightforward and often effective method.*

*- 'median': The final prediction is the median of all individual model
predictions. This can be more robust to outliers in individual
predictions than the mean.*

*- 'weighted_average': The final prediction is a weighted average of the
individual model predictions. Models that performed better during their
training (i.e., had lower errors) are given a higher weight, allowing
more "reliable" experts to influence the final outcome more significantly.*


error_metric : {'RMSE', 'NRMSE', 'NDEI', 'MAE', 'MAPE', 'CPPM'}, default='RMSE'
This is the performance metric used to evaluate and select the best
individual models during the training process. The goal is to minimize 
these error metrics (or maximize CPPM, as it's a "correctness" metric).

*- 'RMSE': Root Mean Squared Error. Penalizes large errors more heavily,
making it sensitive to outliers.*

*- 'NRMSE': Normalized Root Mean Squared Error. RMSE scaled by the range
of the target variable, making it unit-less and easier to compare
across different datasets.*

*- 'NDEI': Non-Dimensional Error Index. Similar to NRMSE but scaled by
the standard deviation of the target variable.*

*- 'MAE': Mean Absolute Error. Represents the average magnitude of the
errors, giving equal weight to all errors. Less sensitive to outliers
than RMSE.*

*- 'MAPE': Mean Absolute Percentage Error. Expresses error as a
percentage, which is often intuitive for business contexts. It can be
problematic with zero or near-zero actual values.*

*- 'CPPM': Correct Percentual Predictions of Movement. Measures the
percentage of times the model correctly predicts the direction of
change (increase or decrease) in the target variable. A higher CPPM
indicates better directional forecasting. For optimization, its
negative value is used as the fitness function.*

parallel_processing : int
This parameter controls whether the training of individual estimators in
the ensemble will be performed in parallel to speed up the process.

*- -1: Utilizes all available CPU cores on your system, maximizing
parallel computation.*

*- 0: Disables parallel processing; training will be performed sequentially.*

*- >0: Uses the exact specified number of CPU cores for parallel execution.
For example, `parallel_processing=4` would use 4 cores.*


Example of R-NMR

    from nfisis.ensemble import R_NMR
    model = R_NMR(n_estimators = 50, error_metric = "MAE", parallel_processing=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


### Random NTSK (R-NTSK)

    from nfisis.ensemble import R_NTSK

Hyperparameters:

n_estimators : int, default=100
The number of individual models (estimators) that will be generated
and combined to form the ensemble. A higher number of estimators
generally leads to a more robust and accurate ensemble but increases
training time. Think of this as how many "experts" you're gathering
to make a final decision.

n_trials : int, default=5
For each estimator in the ensemble, this parameter specifies the
number of attempts (trials) to find the best-performing underlying
model and its optimal feature subset. More trials increase the
chances of discovering a better individual model, but it also means
more computational effort.

combination : {'mean', 'median', 'weighted_average'}, default='mean'
This hyperparameter dictates the technique used to combine the
predictions from all the individual estimators in the ensemble into
a single final prediction.
*- 'mean': The final prediction is the simple average of all individual
model predictions. This is a straightforward and often effective method.*

*- 'median': The final prediction is the median of all individual model
predictions. This can be more robust to outliers in individual
predictions than the mean.*

*- 'weighted_average': The final prediction is a weighted average of the
individual model predictions. Models that performed better during their
training (i.e., had lower errors) are given a higher weight, allowing
more "reliable" experts to influence the final outcome more significantly.*

error_metric : {'RMSE', 'NRMSE', 'NDEI', 'MAE', 'MAPE', 'CPPM'}, default='RMSE'
This is the performance metric used to evaluate and select the best
individual models during the training process. The goal is to minimize
these error metrics (or maximize CPPM, as it's a "correctness" metric).

*- 'RMSE': Root Mean Squared Error. Penalizes large errors more heavily,
making it sensitive to outliers.*

*- 'NRMSE': Normalized Root Mean Squared Error. RMSE scaled by the range
of the target variable, making it unit-less and easier to compare
across different datasets.*

*- 'NDEI': Non-Dimensional Error Index. Similar to NRMSE but scaled by
the standard deviation of the target variable.*

*- 'MAE': Mean Absolute Error. Represents the average magnitude of the
errors, giving equal weight to all errors. Less sensitive to outliers
than RMSE.*

*- 'MAPE': Mean Absolute Percentage Error. Expresses error as a
percentage, which is often intuitive for business contexts. It can be
problematic with zero or near-zero actual values.*

*- 'CPPM': Correct Percentual Predictions of Movement. Measures the
percentage of times the model correctly predicts the direction of
change (increase or decrease) in the target variable. A higher CPPM
indicates better directional forecasting. For optimization, its
negative value is used as the fitness function.*

parallel_processing : int
This parameter controls whether the training of individual estimators in
the ensemble will be performed in parallel to speed up the process.

*- -1: Utilizes all available CPU cores on your system, maximizing
parallel computation.*

*- 0: Disables parallel processing; training will be performed sequentially.*

*- >0: Uses the exact specified number of CPU cores for parallel execution.
For example, `parallel_processing=4` would use 4 cores.*


Example of R-NMR

    from nfisis.ensemble import R_NTSK
    model = R_NTSK(n_estimators = 200, parallel_processing=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

### Extra information

If you want to look closely to the generated rules, you can see the rules typing:

    model.show_rules()

Otherwise, you can see the histogram of the rules by typing:

    model.plot_hist

The fuzzy models are quite fast, but the genetic and ensembles are still a bit slow. If you think you can contribute to this project regarding the code, speed, etc., please, feel free to contact me and to do so.

Code of Conduct:

NFISiS is a library developed by Kaike Alves. Please read the Code of Conduct for guidance.

Call for Contributions:

The project welcomes your expertise and enthusiasm!

Small improvements or fixes are always appreciated. If you are considering larger contributions to the source code, please contact by email first.