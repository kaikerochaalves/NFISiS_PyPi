# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:14:19 2025

@author: Kaike Sa Teles Rocha Alves
@email: kaikerochaalves@outlook.com
"""
# Importing libraries
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from scipy.stats import mode

class BaseNMFIS:
    def __init__(self, fuzzy_operator):
        # Validate `fuzzy_operator`: 'prod', 'max', 'min', 'minmax'
        if fuzzy_operator not in {"prod", "max", "min", "minmax"}:
            raise ValueError("fuzzy_operator must be one of {'prod', 'max', 'min', 'minmax'}.")
        
        # Hyperparameters
        self.fuzzy_operator = fuzzy_operator
        
        # Shared attributes
        self.OutputTrainingPhase = np.array([])
        self.ResidualTrainingPhase = np.array([])
        self.OutputTestPhase = np.array([])
        # Save the inputs of each rule
        self.X_ = []
    
    def get_params(self, deep=True):
        return {'fuzzy_operator': self.fuzzy_operator}

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

    def Gaussian_membership(self, m, x, std):
        # Prevent division by zero
        epsilon = 1e-10
        std = np.maximum(std, epsilon)
        return np.exp(-0.5 * ((m - x) ** 2) / (std ** 2))

    def tau(self, x):
        for row in self.parameters.index:
            if self.fuzzy_operator == "prod":
                tau = np.prod(self.Gaussian_membership(
                    self.parameters.loc[row, 'mean'], x, self.parameters.loc[row, 'std']))
            elif self.fuzzy_operator == "max":
                tau = np.max(self.Gaussian_membership(
                    self.parameters.loc[row, 'mean'], x, self.parameters.loc[row, 'std']))
            elif self.fuzzy_operator == "min":
                tau = np.min(self.Gaussian_membership(
                    self.parameters.loc[row, 'mean'], x, self.parameters.loc[row, 'std']))
            elif self.fuzzy_operator == "minmax":
                tau = (np.min(self.Gaussian_membership(
                    self.parameters.loc[row, 'mean'], x, self.parameters.loc[row, 'std']))
                    * np.max(self.Gaussian_membership(
                    self.parameters.loc[row, 'mean'], x, self.parameters.loc[row, 'std'])))
            self.parameters.at[row, 'tau'] = max(tau, 1e-10)  # Avoid zero values

    def firing_degree(self, x):
        self.tau(x)
        tau_sum = self.parameters['tau'].sum()
        if tau_sum == 0:
            tau_sum = 1 / self.parameters.shape[0]
        self.parameters['firing_degree'] = self.parameters['tau'] / tau_sum
        
class NTSK(BaseNMFIS):
        
    r"""Regression based on New Takagi-Sugeno-Kang.

    The target is predicted by creating rules, composed of fuzzy sets.
    Then, the output is computed as a firing_degreeed average of each local output 
    (output of each rule).

    Read more in the paper https://doi.org/10.1016/j.engappai.2024.108155.


    Parameters
    ----------
    rules : int, default=5
        Number of fuzzy rules will be created.

    lambda1 : float, possible values are in the interval [0,1], default=1
        Defines the forgetting factor for the algorithm to estimate the consequent parameters.
        This parameters is only used when RLS_option is "RLS"

    adaptive_filter : {'RLS', 'wRLS'}, default='RLS'
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

    

    Attributes
    ----------
    


    See Also
    --------
    NMC : New Mamdani Classifier. Implements a new Mamdani approach for classification.
    NMR : New Mamdani Regressor. Implements a new Mamdani approach for regression.

    Notes
    -----
    
    NMC is a specific case of NTSK for classification.

    """
    
    def __init__(self, rules = 5, lambda1 = 1, adaptive_filter = "RLS", fuzzy_operator = "prod", omega = 1000):
        
        super().__init__(fuzzy_operator)  # Chama o construtor da classe BaseNMFIS
        # Validate `rules`: positive integer
        # if not isinstance(rules, int) or rules <= 0:
        if rules <= 0:
            raise ValueError("Rules must be a positive integer.")

        # Validate `lambda1`: [0, 1]
        if not isinstance(lambda1, (float, int)) or not (0 <= lambda1 <= 1):
            raise ValueError("lambda1 must be a float in the interval [0, 1].")

        # Validate `adaptive_filter`: 'RLS' or 'wRLS'
        if adaptive_filter not in {"RLS", "wRLS"}:
            raise ValueError("Adaptive_filter must be either RLS or wRLS.")
            
        # Validate `omega`: positive integer
        if not isinstance(omega, int) or omega <= 0:
            raise ValueError("omega must be a positive integer.")
        
        # Hyperparameters
        self.rules = rules
        self.lambda1 = lambda1
        self.adaptive_filter = adaptive_filter
        self.omega = omega
        
        # Define the rule-based structure
        if self.adaptive_filter == "RLS":
            self.parameters = pd.DataFrame(columns = ['mean', 'std', 'NumObservations'])
            self.parameters_RLS = {}
        if self.adaptive_filter == "wRLS":
            self.parameters = pd.DataFrame(columns = ['mean', 'std', 'P', 'p_vector', 'Theta', 'NumObservations', 'tau', 'firing_degree'])
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
        # Control variables
        self.ymin = 0.
        self.ymax = 0.
        self.region = 0.
        self.last_y = 0.
        

    def get_params(self, deep=True):
        return {
            'rules': self.rules,
            'lambda1': self.lambda1,
            'adaptive_filter': self.adaptive_filter,
            'fuzzy_operator': self.fuzzy_operator,
            'omega': self.omega,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
         
    def fit(self, X, y):
        
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
        
        # Concatenate X with y
        Data = np.hstack((X, y.reshape(-1, 1), np.zeros((X.shape[0], 2))))
        
        # Compute the number of attributes and samples
        m, n = X.shape[1], X.shape[0]
        
        # Vectorized angle calculation
        Data[1:, m + 1] = np.diff(Data[:, m])
        
        # Min and max calculations and region calculation
        self.ymin, self.ymax = Data[:, m + 1].min(), Data[:, m + 1].max()
        self.region = (self.ymax - self.ymin) / self.rules
        
        # Compute the cluster of the inpute
        for row in range(1, n):
            if Data[row, m + 1] < self.ymax:
                rule = int((Data[row, m + 1] - self.ymin) / self.region)
                Data[row, m + 2] = rule
            else:
                rule = int((Data[row, m + 1] - self.ymin) / self.region)
                Data[row, m + 2] = rule - 1
                
        # Create a dataframe from the array
        df = pd.DataFrame(Data)
        empty = []
        
        # Initialize rules vectorized
        for rule in range(self.rules):
            dfnew = df[df[m + 2] == rule]
            if dfnew.empty:
                empty.append(rule)
                # continue
            mean = dfnew.iloc[:, :m].mean().values[:, None]
            self.X_.append(dfnew.iloc[:, :m].values)
            std = np.nan_to_num(dfnew.iloc[:, :m].std().values[:, None], nan=1.0)
            self.initialize_rule(mean, y[0], std, is_first=(rule == 0))
                
        if empty:
            self.parameters.drop(empty, inplace=True, errors='ignore')
        
        # Initialize outputs
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, y[0])
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, (y[0] - y[0]) ** 2)
        
        for k in range(1, n):

            # Prepare the k-th input vector
            x = X[k, :].reshape((1, -1)).T
            xe = np.insert(x.T, 0, 1, axis=1).T
            rule = int(df.loc[k, m + 2])
            
            # Update the rule
            self.rule_update(rule)
            
            # Update the consequent parameters of the rule
            if self.adaptive_filter == "RLS":
                self.RLS(x, y[k], xe)
            elif self.adaptive_filter == "wRLS":
                self.firing_degree(x)
                self.wRLS(x, y[k], xe)
                
            try:
                if self.adaptive_filter == "RLS":
                    # Compute the output based on the most compatible rule
                    Output = xe.T @ self.parameters_RLS['Theta']
                elif self.adaptive_filter == "wRLS":
                    # Compute the output based on the most compatible rule
                    Output = xe.T @ self.parameters.at[rule, 'Theta']
                
                # Store the results
                self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
                self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, (Output - y[k]) ** 2)
            except:
                
                if self.adaptive_filter == "RLS":
                
                    # Call the model with higher lambda 
                    self.inconsistent_lambda(X, y)
                    
                    # Return the results
                    return self.OutputTrainingPhase
                
            if self.adaptive_filter == "RLS":
                if np.isnan(self.parameters_RLS['Theta']).any() or np.isinf(self.ResidualTrainingPhase).any():
                    
                    # Call the model with higher lambda 
                    self.inconsistent_lambda(X, y)
                    
                    # Return the results
                    return self.OutputTrainingPhase
            
        return self.OutputTrainingPhase
            
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
        
        # Prepare the inputs
        X = X.reshape(-1, self.parameters.loc[0, 'mean'].shape[0])
        self.OutputTestPhase = np.array([])
        
        for x in X:
            
            # Prepare the first input vector
            x = x.reshape((1, -1)).T
            
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            
            if self.adaptive_filter == "RLS":
                # Compute the output based on the most compatible rule
                Output = xe.T @ self.parameters_RLS['Theta']
            
            elif self.adaptive_filter == "wRLS":
                
                # Compute the normalized firing degree
                self.firing_degree(x)
            
                # Compute the output
                Output = sum(self.parameters.loc[row, 'firing_degree'] * xe.T @ self.parameters.loc[row, 'Theta'] for row in self.parameters.index)
                
            # Store the output
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
            
        return np.array(self.OutputTestPhase)
        
    def initialize_rule(self, mean, y, std, is_first=False):
        Theta = np.insert(np.zeros(mean.shape[0]), 0, y)[:, None]
        if self.adaptive_filter == "RLS":
            rule_params = {
                'mean': mean,
                'std': std,
                'NumObservations': 1
            }

            if is_first:
                self.parameters = pd.DataFrame([rule_params])
                self.parameters_RLS['P'] = self.omega * np.eye(mean.shape[0] + 1)
                self.parameters_RLS['p_vector'] = np.zeros(Theta.shape)
                self.parameters_RLS['Theta'] = Theta
            else:
                self.parameters = pd.concat([self.parameters, pd.DataFrame([rule_params])], ignore_index=True)
        
        elif self.adaptive_filter == "wRLS":
            rule_params = {
                'mean': mean,
                'P': self.omega * np.eye(mean.shape[0] + 1),
                'p_vector': np.zeros(Theta.shape),
                'Theta': Theta,
                'NumObservations': 1,
                'firing_degree': 0,
                'std': std
            }
            if is_first:
                self.parameters = pd.DataFrame([rule_params])
            else:
                self.parameters = pd.concat([self.parameters, pd.DataFrame([rule_params])], ignore_index=True)

    def rule_update(self, i):
        # Update the number of observations in the rule
        self.parameters.loc[i, 'NumObservations'] = self.parameters.loc[i, 'NumObservations'] + 1
    
    def inconsistent_lambda(self, X, y):
        
        print(f'The lambda1 of {self.lambda1:.2f} is producing inconsistent values. The new value will be set to {0.01+self.lambda1:.2f}')
        
        # Initialize the model
        model = NTSK(rules = self.rules, lambda1 = 0.01 + self.lambda1, adaptive_filter = self.adaptive_filter)
        # Train the model
        self.OutputTrainingPhase = model.fit(X, y)
        
        # Get rule-based structure
        self.parameters = model.parameters
        self.parameters_RLS = model.parameters_RLS
        # Get new lambda1
        self.lambda1 = model.lambda1
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = model.ResidualTrainingPhase
        # Control variables
        self.ymin = model.ymin
        self.ymax = model.ymax
        self.region = model.region
        self.last_y = model.last_y
        
    def RLS(self, x, y, xe):
        """
        Conventional RLS algorithm
        Adaptive Filtering - Paulo S. R. Diniz
        
        Parameters:
            lambda: forgeting factor
    
        """
               
        lambda1 = 1. if self.lambda1 + xe.T @ self.parameters_RLS['P'] @ xe == 0 else self.lambda1
            
        # K is used here just to make easier to see the equation of the covariance matrix
        K = ( self.parameters_RLS['P'] @ xe ) / ( lambda1 + xe.T @ self.parameters_RLS['P'] @ xe )
        self.parameters_RLS['P'] = ( 1 / lambda1 ) * ( self.parameters_RLS['P'] - K @ xe.T @ self.parameters_RLS['P'] )
        self.parameters_RLS['Theta'] = self.parameters_RLS['Theta'] + ( self.parameters_RLS['P'] @ xe ) * (y - xe.T @ self.parameters_RLS['Theta'] )
            

    def wRLS(self, x, y, xe):
        """
        firing_degreeed Recursive Least Square (wRLS)
        An Approach to Online Identification of Takagi-Sugeno Fuzzy Models - Angelov and Filev

        """
        for row in self.parameters.index:
            self.parameters.at[row, 'P'] = self.parameters.loc[row, 'P'] - (( self.parameters.loc[row, 'firing_degree'] * self.parameters.loc[row, 'P'] @ xe @ xe.T @ self.parameters.loc[row, 'P'])/(1 + self.parameters.loc[row, 'firing_degree'] * xe.T @ self.parameters.loc[row, 'P'] @ xe))
            self.parameters.at[row, 'Theta'] = ( self.parameters.loc[row, 'Theta'] + (self.parameters.loc[row, 'P'] @ xe * self.parameters.loc[row, 'firing_degree'] * (y - xe.T @ self.parameters.loc[row, 'Theta'])) )
        


class NewMamdaniRegressor(BaseNMFIS):
    
    r"""Regression based on New Mamdani Regressor.

    The target is predicted by creating rules, composed of fuzzy sets.
    Then, the output is computed as a firing_degreeed average of each local output 
    (output of each rule).


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

    
    Attributes
    ----------
    

    See Also
    --------
    NMC : New Mamdani Classifier. Implements a new Mamdani approach for classification.
    NMR : New Mamdani Regressor. Implements a new Mamdani approach for regression.
    

    Notes
    -----
    
    NMC is a specific case of NMR for classification.

    """
    
    def __init__(self, rules=5, fuzzy_operator='prod'):
        super().__init__(fuzzy_operator)
        if rules <= 0:
            raise ValueError("`rules` must be a positive integer.")
        self.rules = rules
        
        # Models' parameters
        self.parameters = pd.DataFrame(columns=['mean', 'std', 'y', 'NumObservations', 'tau', 'firing_degree'])
         
    def fit(self, X, y):
        
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
            
        # # Prepare the output
        # y = y.reshape(-1,1)
        
        # Concatenate X with y
        Data = np.hstack((X, y.reshape(-1, 1), np.zeros((X.shape[0], 1))))
        
        # Compute the number of attributes
        m = X.shape[1]
        # Compute the number of samples
        n = X.shape[0]
        
        # Compute the width of each interval
        self.ymin = min(Data[:, m])
        self.ymax = max(Data[:, m])
        self.region = ( self.ymax - self.ymin ) / ( self.rules )
        
        # Compute the input rules
        for row in range(1, n):
            if Data[row, m] < self.ymax:
                rule = int( ( Data[row, m] - self.ymin ) / self.region )
                Data[row, m + 1] = rule
            else:
                rule = int( ( Data[row, m] - self.ymin ) / self.region )
                Data[row, m + 1] = rule - 1
        
        # Create a dataframe from the array
        df = pd.DataFrame(Data)
        empty = []
        
        # Initializing the rules
        for rule in range(self.rules):
            dfnew = df[df[m + 1] == rule]
            if dfnew.empty:
                empty.append(rule)
                
            # Compute statistics for mean and standard deviation
            mean = dfnew.iloc[:, :m].mean().values.reshape(-1, 1)
            self.X_.append(dfnew.iloc[:, :m].values)
            std = dfnew.iloc[:, :m].std().values.reshape(-1, 1)
            y_mean = dfnew.iloc[:, m].mean()
            y_std = dfnew.iloc[:, m].std()
            num_obs = len(dfnew.iloc[:, m])
            
            # Handle missing or invalid standard deviation values
            std = np.where(np.isnan(std) | (std == 0.), 1.0, std)
            y_std = 1.0 if np.isnan(y_std) or y_std == 0.0 else y_std
            
            # Initialize the appropriate rule
            self.initialize_rule(y[0], mean, std, y_mean, y_std, num_obs, is_first=(rule == 0))
            
        # Drop empty rules if necessary
        if empty:
            self.parameters.drop(empty, inplace=True, errors="ignore")
        
        self.OutputTrainingPhase = np.array([])
        # Compute the output in the training phase
        for k in range(X.shape[0]):
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T
            # Compute the normalized firing degree
            self.firing_degree(x)
            # Compute the output
            Output = sum( self.parameters['y_mean'] * self.parameters['firing_degree'] ) / sum( self.parameters['firing_degree'] )
            # Store the output
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output )
        # Return the predictions
        return self.OutputTrainingPhase
            
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
        
        # Prepare the inputs
        X = X.reshape(-1, self.parameters.loc[0, 'mean'].shape[0])
        self.OutputTestPhase = np.array([])
        
        for k in range(X.shape[0]):
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T
            # Compute the normalized firing degree
            self.firing_degree(x)
            # Compute the output
            if sum( self.parameters['firing_degree']) == 0:
                Output = 0
            else:
                Output = sum( self.parameters['y_mean'] * self.parameters['firing_degree'] ) / sum( self.parameters['firing_degree'] )
            # Store the output
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output )

        return self.OutputTestPhase
    
    def initialize_rule(self, y, mean, std, y_mean, y_std, num_obs, is_first=False):
        
        if is_first:
            self.parameters = pd.DataFrame([[mean, std, y_mean, y_std, num_obs, np.array([]), 1., 1., 1.]], columns = ['mean', 'std', 'y_mean', 'y_std', 'NumObservations', 'tau', 'firing_degree_min', 'firing_degree_max', 'firing_degree'])
            Output = y
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y)**2)
        
        else:
            NewRow = pd.DataFrame([[mean, std, y_mean, y_std, num_obs, np.array([]), 1., 1., 1.]], columns = ['mean', 'std', 'y_mean', 'y_std', 'NumObservations', 'tau', 'firing_degree_min', 'firing_degree_max', 'firing_degree'])
            self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)
        
class NewMamdaniClassifier(BaseNMFIS):
    
    """Regression based on New Mamdani Regressor.

    The target is predicted by creating rules, composed of fuzzy sets.
    Then, the output is computed as a firing_degreeed average of each local output 
    (output of each rule).


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

    
    Attributes
    ----------
    

    See Also
    --------
    NMC : New Mamdani Classifier. Implements a new Mamdani approach for classification.
    NMR : New Mamdani Regressor. Implements a new Mamdani approach for regression.
    

    Notes
    -----
    
    NMC is a specific case of NMR for classification.

    """
        
    def __init__(self, fuzzy_operator='minmax'):
        
        super().__init__(fuzzy_operator)
        
        # Models' parameters
        self.parameters = pd.DataFrame(columns=['mean', 'std', 'y', 'NumObservations', 'tau', 'firing_degree'])
        
        # Initialize variables
        self.list_unique = None
        self.mapped_values = None
        self.mapping = None
        self.reverse_mapping = None
        
        
    def fit(self, X, y):
        
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if all elements in y are strings
        if all(isinstance(y_, str) for y_ in y):
            # Map the strings to numeric values
            self.mapped_values, self.mapping = self.map_str_to_numeric(y)
            self.list_unique = np.unique(self.mapped_values)  # Unique numeric values
            # Automatically create the reverse mapping
            self.reverse_mapping = {v: k for k, v in self.mapping.items()}
            y = np.array(self.mapped_values)
        # Check if the inputs contain valid numbers
        elif self.is_numeric_and_finite(y):
            self.list_unique = np.unique(y)
        else:
            raise ValueError("Target y contains neither all numeric nor all string values.")
            
        
        # Concatenate the data
        data = np.concatenate((X, y.reshape(-1,1)), axis=1)
        # Create a dataframe with the data
        df = pd.DataFrame(data)
        # Compute the number of columns in the dataframe
        col_df = df.shape[1] - 1
        
        # Compute the number of unique elements in the output and list it
        self.rules = df[col_df].nunique()
        
        # Check if the results is compatible with classification problem
        if self.rules >= df.shape[0]:
            print("There is many different target values, it doesn't look like a classification problem.")
        
        # Compute the parameters of each cluster
        for i in range(self.rules):
            # Filter the data for the current cluster
            cluster_data = df[df[col_df] == self.list_unique[i]].values
            values_X = cluster_data[:, :-1]
            values_y = cluster_data[:, -1]
        
            # Compute the mean and standard deviation of the cluster features
            X_mean = np.mean(values_X, axis=0).reshape(-1, 1)
            X_std = np.std(values_X, axis=0).reshape(-1, 1)
            y_rule = values_y
        
            # Compute the number of observations in the cluster
            num_obs = cluster_data.shape[0]
        
            # Append cluster data and update parameters
            self.X_.append(values_X)
            new_row = {'mean': X_mean, 'std': X_std, 'y': y_rule, 'NumObservations': num_obs}
            self.parameters = pd.concat([self.parameters, pd.DataFrame([new_row])], ignore_index=True)

        # Preallocate space for the outputs for better performance
        # Map the numeric values back to string using the mapping
        self.OutputTrainingPhase = np.zeros(X.shape[0], dtype=object)
    
        # Precompute necessary structures to avoid repeated operations in the loop
        for k, x in enumerate(X):
            # Prepare the input vector
            x = x.reshape(-1, 1)
            
            # Compute the normalized firing degree
            self.firing_degree(x)
            
            # Get the index of the maximum firing degree
            idxmax = self.parameters["firing_degree"].astype(float).idxmax()
            
            # Compute the mode of the output corresponding to the rule with max firing degree
            Output = mode(self.parameters.loc[idxmax, 'y'], keepdims=False).mode
            
            # Store the output in the preallocated array
            self.OutputTrainingPhase[k] = Output
        
        # Check if the original y were string
        if self.reverse_mapping is not None:
            self.OutputTestPhase = [self.reverse_mapping.get(val) for val in self.OutputTestPhase]
        # Return the predictions
        return self.OutputTrainingPhase
     
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
            
        for k in range(X.shape[0]):
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T
            # Compute the normalized firing degree
            self.firing_degree(x)
            # Find the maximum firing_degree degree
            idxmax = self.parameters["firing_degree"].astype(float).idxmax()
            # Compute the output
            Output = st.mode(self.parameters.loc[idxmax,'y'])
            # Store the output
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output )
        
        # Check if the original y were string
        if self.reverse_mapping is not None:
            self.OutputTestPhase = [self.reverse_mapping.get(val) for val in self.OutputTestPhase]
        return self.OutputTestPhase
    
    # Mapping function for string to numeric
    def map_str_to_numeric(self, y):
        unique_values = np.unique(y)
        mapping = {val: idx for idx, val in enumerate(unique_values)}
        mapped_values = [mapping[val] for val in y]
        
        return mapped_values, mapping