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
import os
import math
import random
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Import models
from nfisis.fuzzy import NTSK, NewMamdaniRegressor

class BaseRNFISiS:
    
    r"""This is the Base for the ensemble regression models.

    Parameters
    ----------
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

        - 'mean': The final prediction is the simple average of all individual
          model predictions. This is a straightforward and often effective method.
        - 'median': The final prediction is the median of all individual model
          predictions. This can be more robust to outliers in individual
          predictions than the mean.
        - 'weighted_average': The final prediction is a weighted average of the
          individual model predictions. Models that performed better during their
          training (i.e., had lower errors) are given a higher weight, allowing
          more "reliable" experts to influence the final outcome more significantly.

    error_metric : {'RMSE', 'NRMSE', 'NDEI', 'MAE', 'MAPE', 'CPPM'}, default='RMSE'
        This is the performance metric used to evaluate and select the best
        individual models during the training process. The goal is to minimize
        these error metrics (or maximize CPPM, as it's a "correctness" metric).

        - 'RMSE': Root Mean Squared Error. Penalizes large errors more heavily,
          making it sensitive to outliers.
        - 'NRMSE': Normalized Root Mean Squared Error. RMSE scaled by the range
          of the target variable, making it unit-less and easier to compare
          across different datasets.
        - 'NDEI': Non-Dimensional Error Index. Similar to NRMSE but scaled by
          the standard deviation of the target variable.
        - 'MAE': Mean Absolute Error. Represents the average magnitude of the
          errors, giving equal weight to all errors. Less sensitive to outliers
          than RMSE.
        - 'MAPE': Mean Absolute Percentage Error. Expresses error as a
          percentage, which is often intuitive for business contexts. It can be
          problematic with zero or near-zero actual values.
        - 'CPPM': Correct Percentual Predictions of Movement. Measures the
          percentage of times the model correctly predicts the direction of
          change (increase or decrease) in the target variable. A higher CPPM
          indicates better directional forecasting. For optimization, its
          negative value is used as the fitness function.

    parallel_processing : int
        This parameter controls whether the training of individual estimators in
        the ensemble will be performed in parallel to speed up the process.

        - -1: Utilizes all available CPU cores on your system, maximizing
          parallel computation.
        - 0: Disables parallel processing; training will be performed sequentially.
        - >0: Uses the exact specified number of CPU cores for parallel execution.
          For example, `parallel_processing=4` would use 4 cores.

    """
    
    def __init__(self, n_estimators, n_trials, combination, error_metric, parallel_processing):
        
        # Validate `n_estimators`: positive integer
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer.")
            
        # Validate `n_trials`: positive integer
        if not isinstance(n_trials, int) or n_trials <= 0:
            raise ValueError("n_estimators must be a positive integer.")
            
        if combination not in {"mean", "median", "weighted_average"}:
            raise ValueError('The hyperparameter combination not in list ["mean", "median", "weighted_average"].')
        
        if error_metric not in {"RMSE", "NRMSE", "NDEI", "MAE", "MAPE", "CPPM"}:
            raise ValueError('The hyperparameter error_metric not in list ["RMSE", "NRMSE", "NDEI", "MAE", "MAPE", "CPPM"].')
                    
        if not isinstance(parallel_processing, int):
            raise ValueError("parallel_processing must be an interger greater than -1.")
            
        # Hyperparameters of the genetic algorithm
        # Number of estimators
        self.n_estimators = n_estimators
        # Metric of error
        self.error_metric = error_metric
        # Number of trials
        self.n_trials = n_trials
        # Combination method
        self.combination = combination
        # Parallel processing
        self.parallel_processing = parallel_processing
        
        # Estimators
        self.estimators = []
        self.errors = []
        
        # Shared attributes
        self.parameters = None
        self.y_pred_training = np.array([])
        self.ResidualTrainingPhase = np.array([])
        self.y_pred_test = np.array([])
        # Save the inputs of each rule
        self.X_ = []
    
    def get_params(self, deep=True):
        return {'n_estimators': self.n_estimators,
                'n_trials': self.n_trials,
                'combination': self.combination,
                'error_metric': self.error_metric,
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
    
    def predict(self, X):
        
        # Keep X
        self.X_predict = X
        # for n_estimator in tqdm(range(self.n_estimators), desc="Training"):
        for i in tqdm(range(len(self.best_models)), desc="Predicting"):
            
            # Get correct columns
            selected_cols = self.selected_cols[i]
            # Get X_selected
            X_selected = self.X_predict[:,selected_cols]
            # Run predictions
            _ = self.best_models[i].predict(X_selected)
        
        # Get predicted results
        # Assuming self.best_models is a list of objects and each has an attribute y_pred_training
        output_testing_phases = np.array([model.y_pred_test for model in self.best_models])
        
        # Compute the mean along the rows (axis=0)
        if self.combination == "mean":
            self.y_pred_test = np.mean(output_testing_phases, axis=0)
        elif self.combination == "median":
            self.y_pred_test = np.median(output_testing_phases, axis=0)
        elif self.combination == "weighted_average":
            # Compute the output
            self.y_pred_test = np.average(output_testing_phases, axis=0, weights=self.scores)
        
        return self.y_pred_test
    
    def process_model(self, i):
        """Function to process a single model for parallel execution."""
        selected_cols = self.selected_cols[i]
        X_selected = self.X_predict[:, selected_cols]
        self.best_models[i].predict(X_selected)
    

class R_NTSK(BaseRNFISiS):
    
    r"""Regression based on Random New Takagi-Sugeno-Kang.

    The ensemble R-NTSK represents a robust model for time series forecasting


    """
    
    def __init__(self, n_estimators = 100, n_trials = 5, combination = "mean", error_metric = "RMSE", parallel_processing=0):
        super().__init__(n_estimators, n_trials, combination, error_metric, parallel_processing)  # Chama o construtor da classe BaseNMFIS
        
        # Models` Data
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.errors = []
        self.scores = np.array([])
        self.best_models = []
        self.selected_cols = []
        
    def get_params(self, deep=True):
        # Retrieve parameters from BaseClass and add additional ones
        params = super().get_params(deep=deep)
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
        
        # Store X and y
        self.X = X
        self.y = y
        
        # Define data
        self.X_train = X[:train,]
        self.y_train = y[:train]
        
        self.X_val = X[train:,]
        self.y_val = y[train:]
        
        # Execute in parallel or not
        if self.parallel_processing == 0:
            # Selected cols
            for n_estimator in tqdm(range(self.n_estimators), desc="Training"):
                
                # Look for good models
                lowest_error, best_model, best_selected_cols = self.trials()
                
                # Append the results in the list
                self.errors.append(lowest_error)
                self.best_models.append(best_model)
                self.selected_cols.append(best_selected_cols)
        else:
            if self.parallel_processing == -1:
                max_workers_ = os.cpu_count()
            else:
                max_workers_ = self.parallel_processing
                
            with Pool(processes=min(max_workers_, cpu_count())) as pool:
                results = [pool.apply_async(self.process_estimator, args=(i,)) for i in range(self.n_estimators)]
        
            
                # Collect results
                results = [r.get() for r in results]  # Ensure all tasks complete
    
            # Extract results
            for lowest_error, best_model, best_selected_cols in results:
                self.errors.append(lowest_error)
                self.best_models.append(best_model)
                self.selected_cols.append(best_selected_cols)
        
        # Get fit results
        # Assuming self.best_models is a list of objects and each has an attribute y_pred_training
        output_training_phases = np.array([model.y_pred_training for model in self.best_models])
        
        # Compute the mean along the rows (axis=0)
        if self.combination == "mean":
            self.y_pred_training = np.mean(output_training_phases, axis=0)
        elif self.combination == "median":
            self.y_pred_training = np.median(output_training_phases, axis=0)
        elif self.combination == "weighted_average":
            # Convert errors from list to array
            errors = np.array(self.errors)  # Example list of errors
            # Compute the score
            max_error = np.max(errors) + 1  # Find the maximum error
            self.scores = max_error - errors  # Subtract each error from the maximum
            # Compute the output
            self.y_pred_training = np.average(output_training_phases, axis=0, weights=self.scores)
        
        # Get residuals
        self.ResidualTrainingPhase = (self.y - self.y_pred_training)**2
        # Get the position of the minimum error
        min_idx = min(enumerate(self.errors), key=lambda x: x[1])[0]
        # Get parameters
        self.parameters = self.best_models[min_idx].parameters
        # Get parameters
        self.X_ = self.best_models[min_idx].X_
        
        return self.y_pred_training
    
    def process_estimator(self, _):
        """
        Function to process a single estimator.
        Returns:
            tuple: (lowest_error, best_model, best_selected_cols)
        """
        return self.trials()

    
    def trials(self):
        
        # Initialize the error
        lowest_error = np.inf
        best_model = None
        best_selected_cols = None
        
        # Look for a model with lower error
        for i in range(self.n_trials):
            
            # Run an instance
            error, model, cols, hp = self.iteration()
            
            # Check if the current iteration is better than previously ones
            if error < lowest_error:
                
                # Update the best results
                lowest_error = error
                best_model = model
                best_selected_cols = cols
                best_hp = hp
                
        # Train the best model with all X
        # Define the columns
        X = self.X[:,best_selected_cols]
        y = self.y[:]
        # Train the model
        best_model = NTSK(**best_hp)
        best_model.fit(X, y)
        
        return lowest_error, best_model, best_selected_cols
    
    def iteration(self):
        
        # Initialize model
        model = None
        
        # Generate candidates for the model
        m = self.X_train.shape[1]  # Number of elements in the array
        selected_cols = np.random.randint(0, 2, size=m)
        selected_cols = selected_cols.flatten()
        selected_cols = selected_cols.astype(bool)
        
        
        # Hyperparameters
        rule = random.randrange(1,20)
        adaptive_filter = "wRLS"
        fuzzy_operator = random.choice(["prod", "min", "max", "minmax", "equal"])
        ponder = random.choice([True, False])
        
        if True not in selected_cols:
            
            s = random.randrange(m)
            selected_cols[s] = True
            
        # Define the columns
        X_train = self.X_train[:,selected_cols]
        X_val = self.X_val[:,selected_cols]
        y_train = self.y_train[:]
        y_val = self.y_val[:]
        
        # Initializing the model
        model = NTSK(rules = rule, adaptive_filter = adaptive_filter, fuzzy_operator = fuzzy_operator, ponder = ponder)
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
            return RMSE, model, selected_cols, model.get_params()
        
        if self.error_metric == "NRMSE":
            return NRMSE, model, selected_cols, model.get_params()
        
        if self.error_metric == "NDEI":
            return NDEI, model, selected_cols, model.get_params()
        
        if self.error_metric == "MAE":
            return MAE, model, selected_cols, model.get_params()
        
        if self.error_metric == "MAPE":
            return MAPE, model, selected_cols, model.get_params()
        
        if self.error_metric == "CPPM":
            return -CPPM, model, selected_cols, model.get_params()
    
class R_NMR(BaseRNFISiS):
    
    r"""Regression based on Random New Mamdani Regressor.

    The ensemble R-NMR represents a robust model for time series forecasting

    """
    
    def __init__(self, n_estimators = 100, n_trials = 5, combination = "mean", error_metric = "RMSE", parallel_processing=0):
        super().__init__(n_estimators, n_trials, combination, error_metric, parallel_processing)  # Chama o construtor da classe BaseNMFIS

        # Models` Data
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.errors = []
        self.scores = np.array([])
        self.best_models = []
        self.selected_cols = []
        
    def get_params(self, deep=True):
        # Retrieve parameters from BaseClass and add additional ones
        params = super().get_params(deep=deep)
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
        
        # Store X and y
        self.X = X
        self.y = y
        
        # Define data
        self.X_train = X[:train,]
        self.y_train = y[:train]
        
        self.X_val = X[train:,]
        self.y_val = y[train:]
        
        # Execute in parallel or not
        if self.parallel_processing == 0:
            # Selected cols
            for n_estimator in tqdm(range(self.n_estimators), desc="Training"):
                
                # Look for good models
                lowest_error, best_model, best_selected_cols = self.trials()
                
                # Append the results in the list
                self.errors.append(lowest_error)
                self.best_models.append(best_model)
                self.selected_cols.append(best_selected_cols)
        else:
            if self.parallel_processing == -1:
                max_workers_ = os.cpu_count()
            else:
                max_workers_ = self.parallel_processing
            
            with Pool(processes=min(max_workers_, cpu_count())) as pool:
                results = [pool.apply_async(self.process_estimator, args=(i,)) for i in range(self.n_estimators)]
        
            
                # Collect results
                results = [r.get() for r in results]  # Ensure all tasks complete
    
            # Extract results
            for lowest_error, best_model, best_selected_cols in results:
                self.errors.append(lowest_error)
                self.best_models.append(best_model)
                self.selected_cols.append(best_selected_cols)
        
        # Get fit results
        # Assuming self.best_models is a list of objects and each has an attribute y_pred_training
        output_training_phases = np.array([model.y_pred_training for model in self.best_models])
        
        # Compute the mean along the rows (axis=0)
        if self.combination == "mean":
            self.y_pred_training = np.mean(output_training_phases, axis=0)
        elif self.combination == "median":
            self.y_pred_training = np.median(output_training_phases, axis=0)
        elif self.combination == "weighted_average":
            # Convert errors from list to array
            errors = np.array(self.errors)  # Example list of errors
            # Compute the score
            max_error = np.max(errors) + 1  # Find the maximum error
            self.scores = max_error - errors  # Subtract each error from the maximum
            # Compute the output
            self.y_pred_training = np.average(output_training_phases, axis=0, weights=self.scores)
        
        # Get residuals
        self.ResidualTrainingPhase = (self.y - self.y_pred_training)**2
        # Get the position of the minimum error
        min_idx = min(enumerate(self.errors), key=lambda x: x[1])[0]
        # Get parameters
        self.parameters = self.best_models[min_idx].parameters
        # Get parameters
        self.X_ = self.best_models[min_idx].X_
        
        return self.y_pred_training
    
    def process_estimator(self, _):
        """
        Function to process a single estimator.
        Returns:
            tuple: (lowest_error, best_model, best_selected_cols)
        """
        return self.trials()

    
    def trials(self):
        
        # Initialize the error
        lowest_error = np.inf
        best_model = None
        best_selected_cols = None
        
        # Look for a model with lower error
        for i in range(self.n_trials):
            
            # Run an instance
            error, model, cols, hp = self.iteration()
            
            # Check if the current iteration is better than previously ones
            if error < lowest_error:
                
                # Update the best results
                lowest_error = error
                best_model = model
                best_selected_cols = cols
                best_hp = hp
                
        # Train the best model with all X
        # Define the columns
        X = self.X[:,best_selected_cols]
        y = self.y[:]
        # Train the model
        best_model = NewMamdaniRegressor(**best_hp)
        best_model.fit(X, y)
        
        return lowest_error, best_model, best_selected_cols
    
    def iteration(self):
        
        # Initialize model
        model = None
        
        # Generate candidates for the model
        m = self.X_train.shape[1]  # Number of elements in the array
        selected_cols = np.random.randint(0, 2, size=m)
        selected_cols = selected_cols.flatten()
        selected_cols = selected_cols.astype(bool)
        
        
        # Hyperparameters
        rule = random.randrange(1,20)
        fuzzy_operator = random.choice(["prod", "min", "max", "minmax", "equal"])
        ponder = random.choice([True, False])
        
        if True not in selected_cols:
            
            s = random.randrange(m)
            selected_cols[s] = True
            
        # Define the columns
        X_train = self.X_train[:,selected_cols]
        X_val = self.X_val[:,selected_cols]
        y_train = self.y_train[:]
        y_val = self.y_val[:]
        
        # Initializing the model
        model = NewMamdaniRegressor(rules = rule, fuzzy_operator = fuzzy_operator, ponder = ponder)
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
            return RMSE, model, selected_cols, model.get_params()
        
        if self.error_metric == "NRMSE":
            return NRMSE, model, selected_cols, model.get_params()
        
        if self.error_metric == "NDEI":
            return NDEI, model, selected_cols, model.get_params()
        
        if self.error_metric == "MAE":
            return MAE, model, selected_cols, model.get_params()
        
        if self.error_metric == "MAPE":
            return MAPE, model, selected_cols, model.get_params()
        
        if self.error_metric == "CPPM":
            return -CPPM, model, selected_cols, model.get_params()
