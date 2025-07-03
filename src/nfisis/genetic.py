# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:14:19 2025

@author: Kaike Sa Teles Rocha Alves
@email_1: kaikerochaalves@outlook.com
@email_2: kaike.alves@estudante.ufjf.br

As a Data Science Manager at PGE-PR and a Ph.D. in Computational Modeling
at the Federal University of Juiz de Fora (UFJF), I specialize in artificial
intelligence, focusing on the study, development, and application of machine learning
models, under the supervision of Prof. Eduardo Pestana de Aguiar. My academic journey 
includes a scholarship that allowed me to pursue a year of my Ph.D. at the 
University of Nottingham/UK, where I was a member of the LUCID
(Laboratory for Uncertainty in Data and Decision Making) under the 
supervision of Professor Christian Wagner. My background in Industrial
Engineering provides me with a holistic view of organizational processes,
enabling me to propose efficient solutions and reduce waste.
"""

# Importing libraries
import math
import pygad
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

# Import models
from nfisis.fuzzy import NTSK, NewMamdaniRegressor

class BaseGENNFISiS:
    
    r"""This is the Base for the genetic regression models.

    Parameters
    ----------
    num_generations (int):
    The number of generations or iterations for the model's optimization process. This parameter determines how many cycles the algorithm will run to evolve the solution.
    
    num_parents_mating (int):
    The number of parent solutions selected for mating in each generation. This controls the diversity and combination of solutions to produce the next generation.
    
    sol_per_pop (int):
    The number of solutions in each population. This represents the size of the candidate solution pool used in each generation of the optimization process.
    
    error_metric (str or callable):
    The metric used to evaluate the performance of candidate solutions. This can be a string representing a predefined metric (e.g., "mse" for mean squared error) or a custom function that returns an error score.
    
    print_information (bool):
    A flag indicating whether to display detailed progress information during training. When set to True, the model logs information such as the current generation, fitness scores, and optimization status.
    
    parallel_processing (bool or int):
    Determines whether to use parallel processing for the optimization process. If True, the model utilizes all available CPU cores. If an integer is provided, it specifies the number of cores to use.

    """
    
    def __init__(self, num_generations, num_parents_mating, sol_per_pop, error_metric, print_information, parallel_processing):
        
        if not isinstance(num_generations, int) or not (num_generations>0):
            raise ValueError('The hyperparameter num_generations must be interger greater than 0.')
        
        if not isinstance(num_parents_mating, int) or not (num_parents_mating>0):
            raise ValueError("Hyperparameter num_parents_mating must be interger greater than 0.")
        
        if not isinstance(sol_per_pop, int) or not (sol_per_pop>0):
            raise ValueError('The hyperparameter sol_per_pop must be interger greater than 0.')
        
        if error_metric not in {"RMSE", "NRMSE", "NDEI", "MAE", "MAPE", "CPPM"}:
            raise ValueError('The hyperparameter error_metric not in list ["RMSE", "NRMSE", "NDEI", "MAE", "MAPE", "CPPM"].')
        
        if print_information not in {True, False}:
            raise ValueError("print_information must be True of False.")
            
        if not isinstance(parallel_processing, (int, type(None))) or (parallel_processing is not None and not parallel_processing > 0):
            raise ValueError("parallel_processing must be an interger greater than zero or none.")
            
        # Hyperparameters of the genetic algorithm
        # Metric of error
        self.error_metric = error_metric
        # Number of generations
        self.num_generations = num_generations
        # Number of solutions to be selected as parents in the mating pool
        self.num_parents_mating = num_parents_mating
        # Number of solutions in the population
        self.sol_per_pop = sol_per_pop
        # Print information
        self.print_information = print_information
        # Parallel processing
        self.parallel_processing = parallel_processing
        
        # Shared attributes
        self.parameters = None
        self.y_pred_training = np.array([])
        self.ResidualTrainingPhase = np.array([])
        self.y_pred_test = np.array([])
        # Save the inputs of each rule
        self.X_ = []
    
    def get_params(self, deep=True):
        return {'error_metric': self.error_metric,
                'num_generations': self.num_generations,
                'num_parents_mating': self.num_parents_mating,
                'sol_per_pop': self.sol_per_pop,
                'print_information': self.print_information,
                'parallel_processing': self.parallel_processing
                }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def show_rules(self):
        rules = []
        for i in self.parameters.index:
            rule = f"Rule {i}"
            for j in range(self.parameters.loc[i,"mean"].shape[0]):
                rule = f'{rule} - {self.parameters.loc[i,"mean"][j].item():.2f} ({self.parameters.loc[i,"std"][j].item():.2f})'
            print(rule)
            rules.append(rule)
        
        return rules
    
    def plot_hist(self, bins=10):
        # Set plot-wide configurations only once
        plt.rc('font', size=30)
        plt.rc('axes', titlesize=30)
        
        # Iterate through rules and attributes
        for i, data in enumerate(self.X_):
            for j in range(data.shape[1]):
                # Create and configure the plot
                plt.figure(figsize=(19.20, 10.80))  # Larger figure for better clarity
                plt.hist(
                    data[:, j], 
                    bins=bins, 
                    alpha=0.7,  # Slight transparency for better visuals
                    color='blue', 
                    edgecolor='black'
                )
                # Add labels and titles
                plt.title(f'Rule {i} - Attribute {j}')
                plt.xlabel('Values')
                plt.ylabel('Frequency')
                plt.grid(False)
                
                # Display the plot
                plt.show()
    
    def is_numeric_and_finite(self, array):
        return np.isfinite(array).all() and np.issubdtype(np.array(array).dtype, np.number)
    
    def predict(self, X):
        
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
        
        # Initialize the outputs
        self.y_pred_test = np.array([])
        
        # Selected cols
        selected_cols = self.solution
        selected_cols = selected_cols.flatten()
        selected_cols = selected_cols.astype(bool)
        
        # Define the columns
        X_test = X[:,selected_cols]
        
        # Perform predictions
        self.y_pred_test = self.model.predict(X_test)
        
        return self.y_pred_test
    
    

class GEN_NTSK(BaseGENNFISiS):
    
    r"""Regression based on Genetic New Takagi-Sugeno-Kang.

    The genetic algorithm works as an attribute selector algorithm combined with the ML model.

    Parameters
    ----------
    rules : int, default=5
        Number of fuzzy rules will be created.

    lambda1 : float, possible values are in the interval [0,1], default=1
        Defines the forgetting factor for the algorithm to estimate the consequent parameters.
        This parameters is only used when RLS_option is "RLS"

    adaptive_filter : {'RLS', 'wRLS'}, default='wRLS'
        Algorithm used to compute the consequent parameters:

        - 'RLS' will use :class:`RLS`
        - 'wRLS' will use :class:`wRLS`
    
    fuzzy_operator : {'prod', 'max', 'min'}, default='prod'
        Choose the fuzzy operator:

        - 'prod' will use :`product`
        - 'max' will use :class:`maximum value`
        - 'min' will use :class:`minimum value`
        - 'minmax' will use :class:`minimum value multiplied by maximum`

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
        - 'RMSE': Root Mean Squared Error.
        - 'NRMSE': Normalized Root Mean Squared Error.
        - 'NDEI': Non-Dimensional Error Index.
        - 'MAE': Mean Absolute Error.
        - 'MAPE': Mean Absolute Percentage Error.

    print_information : bool, default=False
        If True, information about the genetic algorithm's progress (e.g., generation number,
        current fitness, and fitness change) will be printed during the `fit` process.

    parallel_processing : list or None, default=None
        Configuration for parallel processing using PyGAD's capabilities.
        Refer to PyGAD's documentation for valid formats. If None, parallel processing
        is not used. 
        - parallel_processing=None: no parallel processing is applied,
        - parallel_processing=['process', 10]: applies parallel processing with 10 processes,
        - parallel_processing=['thread', 5] or parallel_processing=5: applies parallel processing with 5 threads.


    """
    
    def __init__(self, rules = 5, lambda1 = 1, adaptive_filter = "wRLS", fuzzy_operator = "prod", omega = 1000, ponder = True, num_generations = 10, num_parents_mating = 5, sol_per_pop = 10, error_metric = "RMSE", print_information=False, parallel_processing=None):
        super().__init__(num_generations, num_parents_mating, sol_per_pop, error_metric, print_information, parallel_processing)  # Chama o construtor da classe BaseNMFIS
        
        # Hyperparameters
        self.rules = rules
        self.lambda1 = lambda1
        self.adaptive_filter = adaptive_filter
        self.fuzzy_operator = fuzzy_operator
        self.omega = omega
        self.ponder = ponder
        
        # Inferior limit
        self.init_range_low = 0
        # Superior limit 2 (not including)
        self.init_range_high = 2
        
        # Models` Data
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.X_test = np.array([])
        self.y_test = np.array([])
        self.last_fitness = None
        self.model = None
        self.columns = None
        self.selected_cols = None
        
    def get_params(self, deep=True):
        # Retrieve parameters from BaseClass and add additional ones
        params = super().get_params(deep=deep)
        params.update({
            'rules': self.rules,
            'lambda1': self.lambda1,
            'adaptive_filter': self.adaptive_filter,
            'fuzzy_operator': self.fuzzy_operator,
            'omega': self.omega,
            'ponder': self.ponder
        })
        return params

    def set_params(self, **params):
        # Set parameters in BaseClass and the new ones
        super().set_params(**params)
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def fit(self, X, y):
        
        # Separate into train and test
        n = X.shape[0]
        train = round(n * 0.75)
        y = y.ravel()
        # Define data
        self.X_train = X[:train,]
        self.y_train = y[:train]
        
        self.X_val = X[train:,]
        self.y_val = y[train:]
        
        # Number of genes
        num_genes = self.X_train.shape[1]
        
        def on_generation(ga_instance):
            if self.print_information == True:
                print(f"Generation = {ga_instance.generations_completed}")
                print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
                print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - self.last_fitness}")
            self.last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

        ga_instance = pygad.GA(num_generations=self.num_generations,
                               num_parents_mating=self.num_parents_mating,
                               sol_per_pop=self.sol_per_pop,
                               num_genes=num_genes,
                               init_range_low=self.init_range_low,
                               init_range_high=self.init_range_high,
                               fitness_func=self.genetic_algorithm,
                               on_generation=on_generation,
                               suppress_warnings=True,
                               gene_type=int,
                               parallel_processing=self.parallel_processing)
        
        # Running the GA to optimize the parameters of the function.
        ga_instance.run()
        
        # Returning the details of the best solution.
        self.solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        if self.print_information == True:
            print(f"Parameters of the best solution : {self.solution}")
            print(f"Fitness value of the best solution = {solution_fitness}")
            print(f"Index of the best solution : {solution_idx}")
            
            if ga_instance.best_solution_generation != -1:
                print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")
        
        # Saving the GA instance.
        # The filename to which the instance is saved. The name is without extension
        # filename = 'Results_Genetic_Algorithm' 
        # ga_instance.save(filename=filename)
        
        # # Use the next function to load the saved GA instance.
        # loaded_ga_instance = pygad.load(filename=filename)
        # loaded_ga_instance.plot_fitness()
        
        # Selected cols
        selected_cols = self.solution
        selected_cols = selected_cols.flatten()
        selected_cols = selected_cols.astype(bool)
        
        # Define the columns
        X = X[:,selected_cols]
        
        # Initializing the model
        self.model = NTSK(rules = self.rules, lambda1 = self.lambda1, adaptive_filter = self.adaptive_filter, fuzzy_operator = self.fuzzy_operator, omega = self.omega, ponder = self.ponder)
        # Train the model
        self.model.fit(X, y)
        # Get fit results
        self.y_pred_training = self.model.y_pred_training
        # Get residuals
        self.ResidualTrainingPhase = self.model.ResidualTrainingPhase
        # Get parameters
        self.parameters = self.model.parameters
        # Get parameters
        self.X_ = self.model.X_
        
        return self.model
    
    def genetic_algorithm(self, ga_instance, selected_cols, selected_cols_idx):
        
        selected_cols = selected_cols.flatten()
        selected_cols = selected_cols.astype(bool)
        
        if True not in selected_cols:
            RMSE = np.inf
            NRMSE = np.inf
            NDEI = np.inf
            MAE = np.inf
            MAPE = np.inf
            CPPM = -np.inf
            
        else:
            
            # Define the columns
            X_train = self.X_train[:,selected_cols]
            X_val = self.X_train[:,selected_cols]
            y_train = self.y_train[:]
            y_val = self.y_train[:]
            
            # Initializing the model
            model = NTSK(rules = self.rules, lambda1 = self.lambda1, adaptive_filter = self.adaptive_filter, fuzzy_operator = self.fuzzy_operator)
            # Train the model
            model.fit(X_train, y_train)
            # Test the model
            y_pred = model.predict(X_val)
            
            # Calculating the error metrics
            # Compute the Root Mean Square Error
            try:
                RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
            except:
                print(X_train.shape, X_val.shape, y_val.shape, y_pred.shape)
            # Compute the Normalized Root Mean Square Error
            NRMSE = RMSE/(y_val.max() - y_val.min())
            # Compute the Non-Dimensional Error Index
            NDEI= RMSE/st.stdev(np.asarray(y_val, dtype=np.float64))
            # Compute the Mean Absolute Error
            MAE = mean_absolute_error(y_val, y_pred)
            # Compute the Mean Absolute Percentage Error
            MAPE = mean_absolute_percentage_error(y_val, y_pred)
            # Count number of times the model predict a correct increase or decrease
            # Actual variation
            next_y = y_val[1:]
            current_y = y_val[:-1]
            actual_variation = (next_y - current_y) > 0.
            
            # Predicted variation
            next_y_pred = y_pred[1:]
            current_y_pred = y_pred[:-1]
            pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()
    
            # Right?
            correct = actual_variation == pred_variation
            # Correct Percentual Predictions of Movement
            CPPM = (sum(correct).item()/correct.shape[0])*100
        
        if self.error_metric == "RMSE":
            return -RMSE
        
        if self.error_metric == "NRMSE":
            return -NRMSE
        
        if self.error_metric == "NDEI":
            return -NDEI
        
        if self.error_metric == "MAE":
            return -MAE
        
        if self.error_metric == "MAPE":
            return -MAPE
        
        if self.error_metric == "CPPM":
            return CPPM
        
class GEN_NMR(BaseGENNFISiS):
    
    r"""Regression based on Genetic New Mamdani Regressor.

    The genetic algorithm works as an attribute selector algorithm combined with the ML model.

    Parameters
    ----------
    rules : int, default=5
        Number of fuzzy rules will be created.

    fuzzy_operator : {'prod', 'max', 'min'}, default='prod'
        Choose the fuzzy operator:

        - 'prod' will use :`product`
        - 'max' will use :class:`maximum value`
        - 'min' will use :class:`minimum value`
        - 'minmax' will use :class:`minimum value multiplied by maximum`
    
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
        - 'RMSE': Root Mean Squared Error.
        - 'NRMSE': Normalized Root Mean Squared Error.
        - 'NDEI': Non-Dimensional Error Index.
        - 'MAE': Mean Absolute Error.
        - 'MAPE': Mean Absolute Percentage Error.

    print_information : bool, default=False
        If True, information about the genetic algorithm's progress (e.g., generation number,
        current fitness, and fitness change) will be printed during the `fit` process.

    parallel_processing : list or None, default=None
        Configuration for parallel processing using PyGAD's capabilities.
        Refer to PyGAD's documentation for valid formats. If None, parallel processing
        is not used. 
        - parallel_processing=None: no parallel processing is applied,
        - parallel_processing=['process', 10]: applies parallel processing with 10 processes,
        - parallel_processing=['thread', 5] or parallel_processing=5: applies parallel processing with 5 threads.
        

    """
    
    def __init__(self, rules = 5, fuzzy_operator = "prod", ponder = True, num_generations = 10, num_parents_mating = 5, sol_per_pop = 10, error_metric = "RMSE", print_information=False, parallel_processing=None):
        super().__init__(num_generations, num_parents_mating, sol_per_pop, error_metric, print_information, parallel_processing)  # Chama o construtor da classe BaseNMFIS
        
        # Hyperparameters
        self.rules = rules
        self.fuzzy_operator = fuzzy_operator
        self.ponder = ponder
        
        # Inferior limit
        self.init_range_low = 0
        # Superior limit 2 (not including)
        self.init_range_high = 2
        
        # Models` Data
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.X_test = np.array([])
        self.y_test = np.array([])
        self.last_fitness = None
        self.model = None
        self.columns = None
        self.selected_cols = None
    
    def get_params(self, deep=True):
        # Retrieve parameters from BaseClass and add additional ones
        params = super().get_params(deep=deep)
        params.update({
            'rules': self.rules,
            'fuzzy_operator': self.fuzzy_operator,
            'ponder': self.ponder
        })
        return params

    def set_params(self, **params):
        # Set parameters in BaseClass and the new ones
        super().set_params(**params)
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def fit(self, X, y):
        
        # Separate into train and test
        n = X.shape[0]
        train = round(n * 0.75)
        y = y.ravel()
        # Define data
        self.X_train = X[:train,]
        self.y_train = y[:train]
        
        # Number of genes
        num_genes = self.X_train.shape[1]
        
        def on_generation(ga_instance):
            if self.print_information == True:
                print(f"Generation = {ga_instance.generations_completed}")
                print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
                print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - self.last_fitness}")
            self.last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

        ga_instance = pygad.GA(num_generations=self.num_generations,
                               num_parents_mating=self.num_parents_mating,
                               sol_per_pop=self.sol_per_pop,
                               num_genes=num_genes,
                               init_range_low=self.init_range_low,
                               init_range_high=self.init_range_high,
                               fitness_func=self.genetic_algorithm,
                               on_generation=on_generation,
                               suppress_warnings=True,
                               gene_type=int,
                               parallel_processing=self.parallel_processing)
        
        # Running the GA to optimize the parameters of the function.
        ga_instance.run()
        
        # Returning the details of the best solution.
        self.solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        if self.print_information == True:
            print(f"Parameters of the best solution : {self.solution}")
            print(f"Fitness value of the best solution = {solution_fitness}")
            print(f"Index of the best solution : {solution_idx}")
            
            if ga_instance.best_solution_generation != -1:
                print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")
        
        # # Saving the GA instance.
        # # The filename to which the instance is saved. The name is without extension
        # filename = 'Results_Genetic_Algorithm' 
        # ga_instance.save(filename=filename)
                
        # Selected cols
        selected_cols = self.solution
        selected_cols = selected_cols.flatten()
        selected_cols = selected_cols.astype(bool)
        
        # Define the columns
        X = X[:,selected_cols]
        
        # Initializing the model
        self.model = NewMamdaniRegressor(rules = self.rules, fuzzy_operator = self.fuzzy_operator, ponder = self.ponder)
        # Train the model
        self.model.fit(X, y)
        # Get fit results
        self.y_pred_training = self.model.y_pred_training
        # Get residuals
        self.ResidualTrainingPhase = self.model.ResidualTrainingPhase
        # Get parameters
        self.parameters = self.model.parameters
        # Get parameters
        self.X_ = self.model.X_
        
        return self.model
    
    def genetic_algorithm(self, ga_instance, selected_cols, selected_cols_idx):
        
        selected_cols = selected_cols.flatten()
        selected_cols = selected_cols.astype(bool)
        
        if True not in selected_cols:
            RMSE = np.inf
            NRMSE = np.inf
            NDEI = np.inf
            MAE = np.inf
            MAPE = np.inf
            CPPM = -np.inf
            
        else:
            
            # Define the columns
            X_train = self.X_train[:,selected_cols]
            X_val = self.X_train[:,selected_cols]
            y_train = self.y_train[:]
            y_val = self.y_train[:]
            
            # Initializing the model
            model = NewMamdaniRegressor(rules = self.rules, fuzzy_operator = self.fuzzy_operator)
            # Train the model
            model.fit(X_train, y_train)
            # Test the model
            y_pred = model.predict(X_val)
            
            # Calculating the error metrics
            # Compute the Root Mean Square Error
            RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
            # Compute the Normalized Root Mean Square Error
            NRMSE = RMSE/(y_val.max() - y_val.min())
            # Compute the Non-Dimensional Error Index
            NDEI= RMSE/st.stdev(np.asarray(y_val, dtype=np.float64))
            # Compute the Mean Absolute Error
            MAE = mean_absolute_error(y_val, y_pred)
            # Compute the Mean Absolute Percentage Error
            MAPE = mean_absolute_percentage_error(y_val, y_pred)
            # Count number of times the model predict a correct increase or decrease
            # Actual variation
            next_y = y_val[1:]
            current_y = y_val[:-1]
            actual_variation = (next_y - current_y) > 0.
            
            # Predicted variation
            next_y_pred = y_pred[1:]
            current_y_pred = y_pred[:-1]
            pred_variation = ((next_y_pred - current_y_pred) > 0.).flatten()
    
            # Right?
            correct = actual_variation == pred_variation
            # Correct Percentual Predictions of Movement
            CPPM = (sum(correct).item()/correct.shape[0])*100
        
        if self.error_metric == "RMSE":
            return -RMSE
        
        if self.error_metric == "NRMSE":
            return -NRMSE
        
        if self.error_metric == "NDEI":
            return -NDEI
        
        if self.error_metric == "MAE":
            return -MAE
        
        if self.error_metric == "MAPE":
            return -MAPE
        
        if self.error_metric == "CPPM":
            return CPPM