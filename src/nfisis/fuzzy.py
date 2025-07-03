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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

class BaseNMFISiS:
    def __init__(self, fuzzy_operator, ponder):
        
        r"""This is the Base for the new fuzzy inference systems
        """
        # Validate `fuzzy_operator`: 'prod', 'max', 'min', 'minmax', 'equal'
        if fuzzy_operator not in {"prod", "max", "min", "minmax", "equal"}:
            raise ValueError("fuzzy_operator must be one of {'prod', 'max', 'min', 'minmax', 'equal'}.")
        # Validate `pond`: True or False
        if not isinstance(ponder, bool):
            raise ValueError("`ponder` must be a boolean.")
        
        # Hyperparameters
        self.fuzzy_operator = fuzzy_operator
        # Ponder
        self.ponder = ponder
        
        # Shared attributes
        self.y_pred_training = None
        self.ResidualTrainingPhase = None
        self.y_pred_test = None
        # Save the inputs of each rule
        self.X_ = []
    
    def get_params(self, deep=True):
        return {'fuzzy_operator': self.fuzzy_operator,
                'ponder': self.ponder
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

    def Gaussian_membership(self, m, x, std):
        # Prevent division by zero
        epsilon = 1e-10
        std = np.maximum(std, epsilon)
        return np.exp(-0.5 * ((m - x) ** 2) / (std ** 2))
        
class NTSK(BaseNMFISiS):
        
    r"""Regression based on New Takagi-Sugeno-Kang.

    The target is predicted by creating rules, composed of fuzzy sets.
    Then, the output is computed as a firing_degreee average of each local output 
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
    
    fuzzy_operator : {'prod', 'max', 'min', 'minmax'}, default='prod'
        Choose the fuzzy operator:

        - 'prod' will use :`product`
        - 'max' will use :class:`maximum value`
        - 'min' will use :class:`minimum value`
        - 'minmax' will use :class:`minimum value multiplied by maximum`

    omega : int, default=1000
        Omega is a parameters used to initialize the algorithm to estimate
        the consequent parameters

    ponder : boolean, default=True
        ponder controls whether the firing degree of each fuzzy rule 
        is weighted by the number of observations (data points) 
        associated with that rule during the tau calculation.
        Used to avoid the influence of less representative rules
        
    
    See Also
    --------
    NMC : New Mamdani Classifier. Implements a new Mamdani approach for classification.
    NMR : New Mamdani Regressor. Implements a new Mamdani approach for regression.

    Notes
    -----
    
    NMC is a specific case of NTSK for classification.

    """
    
    def __init__(self, rules = 5, lambda1 = 1, adaptive_filter = "RLS", fuzzy_operator = "prod", omega = 1000, ponder = True):
        
        super().__init__(fuzzy_operator, ponder)  # Chama o construtor da classe BaseNMFISiS
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
            self.parameters_list = []
            self.parameters_RLS_list = []
        if self.adaptive_filter == "wRLS":
            self.parameters_list = []
            
        # Control variables
        self.ymin = 0.
        self.ymax = 0.
        self.region = 0.
        self.last_y = 0.
        

    def get_params(self, deep=True):
        # Retrieve parameters from BaseClass and add additional ones
        params = super().get_params(deep=deep)
        params.update({
            'rules': self.rules,
            'lambda1': self.lambda1,
            'adaptive_filter': self.adaptive_filter,
            'omega': self.omega,
        })
        return params

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
         
    def fit(self, X, y):
        
        # Shape of X and y
        X_shape = X.shape
        y_shape = y.shape
        
        # Correct format X to 2d
        if len(X_shape) == 1:
            
            # Reshape X
            X = X.reshape(-1,1)
            
            # Get new shape of X
            X_shape = X.shape
        
        # Check wheather y is 1d
        if len(y_shape) > 1 and y_shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y_shape) > 1:
            
            # 1d
            y = y.ravel()
            
            # Get new shape of y
            y_shape = y.shape
        
        # Check wheather y is 1d
        if X_shape[0] != y_shape[0]:
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
        Data = np.hstack((X, y.reshape(-1, 1), np.zeros((X_shape[0], 2))))
        
        # Compute the number of attributes and samples
        m, n = X_shape[1], X_shape[0]
        
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
        
        # Initialize rules vectorized
        for rule in range(self.rules):
            dfnew = df[df[m + 2] == rule]
            if not dfnew.empty:
                mean = dfnew.iloc[:, :m].mean().values[:, None]
                self.X_.append(dfnew.iloc[:, :m].values)
                std = np.nan_to_num(dfnew.iloc[:, :m].std().values[:, None], nan=1.0)
                self.initialize_rule(mean, y[0], std, dfnew.shape[0], is_first=(rule == 0))
        
        # Preallocate space for the outputs for better performance
        self.y_pred_training = np.zeros((y_shape))
        self.ResidualTrainingPhase = np.zeros((y_shape))
        # Initialize outputs
        self.y_pred_training[0,] = y[0]
        self.ResidualTrainingPhase[0,] = 0.
        
        # Check the adaptive_filter
        if self.adaptive_filter == "RLS":
            
            for k in range(1, n):
    
                # Prepare the k-th input vector
                x = X[k, :].reshape((1, -1)).T
                xe = np.insert(x.T, 0, 1, axis=1).T
                rule = int(df.loc[k, m + 2])
                
                # Update the consequent parameters of the rule
                self.RLS(x, y[k], xe)
                    
                try:
                    
                    # Compute the output based on the most compatible rule
                    Output = xe.T @ self.parameters_RLS_list[0][1]
                    
                    # Store the results
                    self.y_pred_training[k,] = Output.item()
                    self.ResidualTrainingPhase[k,] = (Output - y[k]) ** 2
                    
                except:
                
                    # Call the model with higher lambda 
                    self.inconsistent_lambda(X, y)
                    
                    # Return the results
                    return self.y_pred_training

                if np.isnan(self.parameters_RLS_list[0][1]).any() or np.isinf(self.ResidualTrainingPhase).any():
                    
                    # Call the model with higher lambda 
                    self.inconsistent_lambda(X, y)
                    
                    # Return the results
                    return self.y_pred_training
                
            # Save the rules to a dataframe
            self.parameters = pd.DataFrame(self.parameters_list, columns=["mean", "std", "NumObservations"])
            self.parameters_RLS = pd.DataFrame(self.parameters_RLS_list, columns=["P", "Theta"])
                    
        elif self.adaptive_filter == "wRLS":
            
            for k in range(1, n):
    
                # Prepare the k-th input vector
                x = X[k, :].reshape((1, -1)).T
                xe = np.insert(x.T, 0, 1, axis=1).T
                
                # Define the rules
                rule = int(df.loc[k, m + 2])
                
                # # Update the rule
                # self.rule_update(rule)
                
                # Update the consequent parameters of the rule
                self.firing_degree(x)
                self.wRLS(x, y[k], xe)
                
                # Compute the output based on the most compatible rule
                # Compute the output
                Output = sum(row[6] * xe.T @ row[1] for row in self.parameters_list)
                
                # Store the results
                self.y_pred_training[k,] = Output.item()
                self.ResidualTrainingPhase[k,] = (Output - y[k]) ** 2
            
            # Save the rules to a dataframe
            self.parameters = pd.DataFrame(self.parameters_list, columns=["P", "Theta", "mean", "std", "NumObservations", "tau", "firing_degree"])

        return self.y_pred_training
            
    def predict(self, X):
        
        # Shape of X
        X_shape = X.shape
        
        # Correct format X to 2d
        if len(X_shape) == 1:
            
            # Reshape X
            X = X.reshape(-1,1)
            
            # Get new shape of X
            X_shape = X.shape
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
        
        # Prepare the inputs
        X = X.reshape(-1, self.parameters.loc[0, 'mean'].shape[0])
        
        # Preallocate space for the outputs for better performance
        self.y_pred_test = np.zeros((X_shape[0],))
        
        if self.adaptive_filter == "RLS":
            
            for k in range(X_shape[0]):
                
                # Prepare the first input vector
                x = X[k,].reshape((1,-1)).T
                
                # Compute xe
                xe = np.insert(x.T, 0, 1, axis=1).T
                
                # Compute the output based on the most compatible rule
                Output = xe.T @ self.parameters_RLS_list[0][1]
                
                # Store the output
                self.y_pred_test[k,] = Output.item()
        
        elif self.adaptive_filter == "wRLS":
            
            for k in range(X_shape[0]):
                
                # Prepare the first input vector
                x = X[k,].reshape((1,-1)).T
                
                # Compute xe
                xe = np.insert(x.T, 0, 1, axis=1).T
                
                # Compute the normalized firing degree
                self.firing_degree(x)
            
                # Compute the output
                Output = sum(row[6] * xe.T @ row[1] for row in self.parameters_list)
                
                # Store the output
                self.y_pred_test[k,] = Output.item()
            
        return np.array(self.y_pred_test)
        
    def initialize_rule(self, mean, y, std, num_obs, is_first=False):
        Theta = np.insert(np.zeros(mean.shape[0]), 0, y)[:, None]
        
        if self.adaptive_filter == "RLS":
                        
            # Include the rules in a list - 0: mean, 1: std, 2: NumObservations
            self.parameters_list.append([mean, std, num_obs])

            if is_first:
                
                # Include the rules in a list - 0: mean, 1: std, 2: NumObservations
                self.parameters_RLS_list.append([self.omega * np.eye(mean.shape[0] + 1), Theta])
        
        elif self.adaptive_filter == "wRLS":
            
            # Include the rules in a list - 0: P, 1: Theta, 2: mean, 3: std, 4: NumObservations, 5: tau, 6: firing_degree
            self.parameters_list.append([self.omega * np.eye(mean.shape[0] + 1), Theta, mean, std, num_obs, 0, 0])
    
    def inconsistent_lambda(self, X, y):
        
        print(f'The lambda1 of {self.lambda1:.2f} is producing inconsistent values. The new value will be set to {0.01+self.lambda1:.2f}')
        
        # Initialize the model
        model = NTSK(rules = self.rules, lambda1 = 0.01 + self.lambda1, adaptive_filter = self.adaptive_filter)
        # Train the model
        self.y_pred_training = model.fit(X, y)
        
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
        
    def tau(self, x):
        
        # Variable to sum tau
        sum_tau = 0
        for row in range(len(self.parameters_list)):
            if self.fuzzy_operator == "prod":
                tau = np.prod(self.Gaussian_membership(
                    self.parameters_list[row][2], x,  self.parameters_list[row][3]))
            elif self.fuzzy_operator == "max":
                tau = np.max(self.Gaussian_membership(
                    self.parameters_list[row][2], x,  self.parameters_list[row][3]))
            elif self.fuzzy_operator == "min":
                tau = np.min(self.Gaussian_membership(
                    self.parameters_list[row][2], x,  self.parameters_list[row][3]))
            elif self.fuzzy_operator == "minmax":
                tau = (np.min(self.Gaussian_membership(
                    self.parameters_list[row][2], x,  self.parameters_list[row][3]))
                    * np.max(self.Gaussian_membership(
                    self.parameters_list[row][2], x,  self.parameters_list[row][3])))
            
            elif self.fuzzy_operator == "equal":
                tau = 1
            
            # Check if it is necessary to multiply tau by the number of observations
            if self.ponder == True:
                tau *= self.parameters_list[row][4]
                
            self.parameters_list[row][5] = max(tau, 1e-10)  # Avoid zero values
            sum_tau += max(tau, 1e-10)
        
        return sum_tau

    def firing_degree(self, x):
        
        # Initialize the total sum
        sum_tau = self.tau(x)
        
        if sum_tau == 0:
            sum_tau = 1 / self.parameters.shape[0]
        for row in range(len(self.parameters_list)):
            self.parameters_list[row][6] = self.parameters_list[row][5] / sum_tau
        
    def RLS(self, x, y, xe):
        """
        Conventional RLS algorithm
        Adaptive Filtering - Paulo S. R. Diniz
        
        Parameters:
            lambda: forgeting factor
    
        """
        
        lambda1 = 1. if self.lambda1 + xe.T @ self.parameters_RLS_list[0][0] @ xe == 0 else self.lambda1
            
        # K is used here just to make easier to see the equation of the covariance matrix
        K = ( self.parameters_RLS_list[0][0] @ xe ) / ( lambda1 + xe.T @ self.parameters_RLS_list[0][0] @ xe )
        self.parameters_RLS_list[0][0] = ( 1 / lambda1 ) * ( self.parameters_RLS_list[0][0] - K @ xe.T @ self.parameters_RLS_list[0][0] )
        self.parameters_RLS_list[0][1] = self.parameters_RLS_list[0][1] + ( self.parameters_RLS_list[0][0] @ xe ) * (y - xe.T @ self.parameters_RLS_list[0][1] )
            

    def wRLS(self, x, y, xe):
        """
        firing_degreeed Recursive Least Square (wRLS)
        An Approach to Online Identification of Takagi-Sugeno Fuzzy Models - Angelov and Filev

        """
        for row in range(len(self.parameters_list)):
            # self.parameters.at[row, 'P'] = self.parameters.loc[row, 'P'] - (( self.parameters.loc[row, 'firing_degree'] * self.parameters.loc[row, 'P'] @ xe @ xe.T @ self.parameters.loc[row, 'P'])/(1 + self.parameters.loc[row, 'firing_degree'] * xe.T @ self.parameters.loc[row, 'P'] @ xe))
            # self.parameters.at[row, 'Theta'] = ( self.parameters.loc[row, 'Theta'] + (self.parameters.loc[row, 'P'] @ xe * self.parameters.loc[row, 'firing_degree'] * (y - xe.T @ self.parameters.loc[row, 'Theta'])) )
            
            firing_degree = self.parameters_list[row][6]
            self.parameters_list[row][0] = self.parameters_list[row][0] - (( firing_degree * self.parameters_list[row][0] @ xe @ xe.T @ self.parameters_list[row][0])/(1 + firing_degree * xe.T @ self.parameters_list[row][0] @ xe))
            self.parameters_list[row][1] = ( self.parameters_list[row][1] + (self.parameters_list[row][0] @ xe * firing_degree * (y - xe.T @ self.parameters_list[row][1])) )
        


class NewMamdaniRegressor(BaseNMFISiS):
    
    r"""Regression based on New Mamdani Regressor.

    The target is predicted by creating rules, composed of fuzzy sets.
    Then, the output is computed as a firing_degreee average of each local output 
    (output of each rule).


    Parameters
    ----------
    rules : int, default=5
        Number of fuzzy rules that will be created.

    
    fuzzy_operator : {'prod', 'max', 'min', 'equal'}, default='prod'
        Choose the fuzzy operator:

        - 'prod' will use :`product`
        - 'max' will use :class:`maximum value`
        - 'min' will use :class:`minimum value`
        - 'minmax' will use :class:`minimum value multiplied by maximum`
        - 'equal' use the same firing degree for all rules
    
    ponder : boolean, default=True
        ponder controls whether the firing degree of each fuzzy rule 
        is weighted by the number of observations (data points) 
        associated with that rule during the tau calculation.
        Used to avoid the influence of less representative rules
        
    See Also
    --------
    NMC : New Mamdani Classifier. Implements a new Mamdani approach for classification.
    NTSK : New Takagi-Sugeno-Kang. Implements a new Takagi-Sugeno-Kang approach for regression.
    

    Notes
    -----
    
    NMC is a specific case of NMR for classification.

    """
    
    def __init__(self, rules=5, fuzzy_operator='prod', ponder = True):
        super().__init__(fuzzy_operator, ponder)
        if rules <= 0:
            raise ValueError("`rules` must be a positive integer.")
        
        # Number of rules
        self.rules = rules
        
        # Parameters
        self.parameters_list = []
         
    def fit(self, X, y):
        
        # Shape of X and y
        X_shape = X.shape
        y_shape = y.shape
        
        # Correct format X to 2d
        if len(X_shape) == 1:
            
            # Reshape X
            X = X.reshape(-1,1)
            
            # Get the new shape of X
            X_shape = X.shape
        
        # Check wheather y is 1d
        if len(y_shape) > 1 and y_shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y_shape) > 1:
            
            # 1d
            y = y.ravel()
            
            # Get new shape of y
            y_shape = y.shape
        
        # Check wheather y is 1d
        if X_shape[0] != y_shape[0]:
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
        Data = np.hstack((X, y.reshape(-1, 1), np.zeros((X_shape[0], 1))))
        
        # Compute the number of attributes
        m = X_shape[1]
        # Compute the number of samples
        n = X_shape[0]
        
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
        
        # Initializing the rules
        for rule in range(self.rules):
            dfnew = df[df[m + 1] == rule]
            
            if not dfnew.empty:
                # empty.append(rule)
                
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
        
        # Preallocate space for the outputs for better performance
        self.y_pred_training = np.zeros((y_shape))
        self.ResidualTrainingPhase = np.zeros((y_shape))
        
        # Compute the output in the training phase
        for k in range(X_shape[0]):
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T
            # Compute the normalized firing degree
            self.firing_degree(x)
            # Compute the output
            Output = sum(row[2] * row[6] for row in self.parameters_list)
            # Store the output
            self.y_pred_training[k,] = Output.item()
            # Store the output
            self.ResidualTrainingPhase[k,] = (Output - y[k]) ** 2
            
        # Save the rules to a dataframe
        self.parameters = pd.DataFrame(self.parameters_list, columns=['mean', 'std', 'y_mean', 'y_std', 'NumObservations', 'tau', 'firing_degree'])
        
        # Return the predictions
        return self.y_pred_training
            
    def predict(self, X):
        
        # Shape of X
        X_shape = X.shape
        
        # Correct format X to 2d
        if len(X_shape) == 1:
            
            # Reshape X
            X = X.reshape(-1,1)
            
            # Shape of X
            X_shape = X.shape
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
        
        # Prepare the inputs
        X = X.reshape(-1, self.parameters.loc[0, 'mean'].shape[0])
        
        # Preallocate space for the outputs for better performance
        self.y_pred_test = np.zeros((X_shape[0],))
        
        for k in range(X_shape[0]):
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T
            # Compute the normalized firing degree
            self.firing_degree(x)
            # Compute the output
            Output = sum(row[2] * row[6] for row in self.parameters_list)
            # Store the output
            self.y_pred_test[k,] = Output.item()

        return self.y_pred_test
    
    def initialize_rule(self, y, mean, std, y_mean, y_std, num_obs, is_first=False):
        
        # Create a list with the parameters
        self.parameters_list.append([mean, std, y_mean, y_std, num_obs, 0, 1.])
        
        if is_first:
            Output = y
            self.y_pred_training = np.append(self.y_pred_training, Output)
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y)**2)
            
    def tau(self, x):
        
        # Variable to sum tau
        sum_tau = 0
        for row in range(len(self.parameters_list)):
            if self.fuzzy_operator == "prod":
                tau = np.prod(self.Gaussian_membership(
                    self.parameters_list[row][0], x,  self.parameters_list[row][1]))
            elif self.fuzzy_operator == "max":
                tau = np.max(self.Gaussian_membership(
                    self.parameters_list[row][0], x,  self.parameters_list[row][1]))
            elif self.fuzzy_operator == "min":
                tau = np.min(self.Gaussian_membership(
                    self.parameters_list[row][0], x,  self.parameters_list[row][1]))
            elif self.fuzzy_operator == "minmax":
                tau = (np.min(self.Gaussian_membership(
                    self.parameters_list[row][0], x,  self.parameters_list[row][1]))
                    * np.max(self.Gaussian_membership(
                    self.parameters_list[row][0], x,  self.parameters_list[row][1])))
            elif self.fuzzy_operator == "equal":
                tau = 1
            
            # Check if it is necessary to multiply tau by the number of observations
            if self.ponder == True:
                tau *= self.parameters_list[row][4]
                
            self.parameters_list[row][5] = max(tau, 1e-10)  # Avoid zero values
            sum_tau += max(tau, 1e-10)
        
        return sum_tau

    def firing_degree(self, x):
        
        # Initialize the total sum
        sum_tau = self.tau(x)
        
        if sum_tau == 0:
            sum_tau = 1 / self.parameters.shape[0]
        for row in range(len(self.parameters_list)):
            self.parameters_list[row][6] = self.parameters_list[row][5] / sum_tau
    
            
class NewMamdaniClassifier(BaseNMFISiS):
    
    """Regression based on New Mamdani Classifier.

    The class is predicted by creating rules, composed of fuzzy sets.
    Then, the output is computed as a firing_degree average of each local output 
    (output of each rule).


    Parameters
    ----------
    rules : int, default=5
        Number of fuzzy rules that will be created.

    
    fuzzy_operator : {'prod', 'max', 'min', 'minmax'}, default='prod'
        Choose the fuzzy operator:

        - 'prod' will use :`product`
        - 'max' will use :class:`maximum value`
        - 'min' will use :class:`minimum value`
        - 'minmax' will use :class:`minimum value multiplied by maximum`


    See Also
    --------
    NTSK : New Takagi-Sugeno-Kang. Implements a new Takagi-Sugeno-Kang approach for regression.
    NMR : New Mamdani Regressor. Implements a new Mamdani approach for regression.

    """
        
    def __init__(self, fuzzy_operator='prod', ponder=True):
        
        super().__init__(fuzzy_operator, ponder)
        
        # Models' parameters
        self.parameters_list = []
        
        # Initialize variables
        self.list_unique = None
        self.mapped_values = None
        self.mapping = None
        self.reverse_mapping = None
        self.dtype = None
        
        
    def fit(self, X, y):
        
        # Shape of X and y
        X_shape = X.shape
        y_shape = y.shape
        
        # Correct format X to 2d
        if len(X_shape) == 1:
            
            # Reshape X
            X = X.reshape(-1,1)
            
            # Get new shape of X
            X_shape = X.shape
        
        # Check wheather y is 1d
        if len(y_shape) > 1 and y_shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y_shape) > 1:
            
            # 1d
            y = y.ravel()
            
            # Get new shape of y
            y_shape = y.shape
        
        # Check wheather y is 1d
        if X_shape[0] != y_shape[0]:
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
        
        # Store y dtype
        self.dtype = y.dtype
        
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
            self.parameters_list.append([X_mean, X_std, y_rule, num_obs, 0., 0.])

        # Preallocate space for the outputs for better performance
        # Map the numeric values back to string using the mapping
        self.y_pred_training = np.zeros(X_shape[0], dtype=self.dtype)
    
        # Precompute necessary structures to avoid repeated operations in the loop
        for k, x in enumerate(X):
            
            # Prepare the input vector
            x = x.reshape(-1, 1)
            
            # Compute the normalized firing degree
            self.firing_degree(x)
            
            # Find the maximum firing_degree degree
            max_firing = 0
            idxmax = 0
            for row in range(len(self.parameters_list)):
                if self.parameters_list[row][5] > max_firing:
                    max_firing = self.parameters_list[row][5]
                    idxmax = row
            
            # Compute the mode of the output corresponding to the rule with max firing degree
            Output = mode(self.parameters_list[idxmax][2], keepdims=False).mode
            
            # Store the output in the preallocated array
            self.y_pred_training[k] = Output
        
        # Check if the original y were string
        if self.reverse_mapping is not None:
            self.y_pred_training = [self.reverse_mapping.get(val) for val in self.y_pred_training]
        
        # Save parameters to a dataframe
        self.parameters = pd.DataFrame(self.parameters_list, columns=['mean', 'std', 'y', 'NumObservations', 'tau', 'firing_degree'])
        
        # Return the predictions
        return self.y_pred_training
     
    def predict(self, X):
        
        # Shape of X
        X_shape = X.shape
        
        # Correct format X to 2d
        if len(X_shape) == 1:
            
            # Reshape X
            X = X.reshape(-1,1)
            
            # Get new shape of X
            X_shape = X.shape
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
        
        # Preallocate space for the outputs for better performance
        # Map the numeric values back to string using the mapping
        self.y_pred_test = np.zeros(X_shape[0], dtype=self.dtype)
        
        for k in range(X_shape[0]):
            
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T
            
            # Compute the normalized firing degree
            self.firing_degree(x)
            
            # Find the maximum firing_degree degree
            max_firing = 0
            idxmax = 0
            for row in range(len(self.parameters_list)):
                if self.parameters_list[row][5] > max_firing:
                    max_firing = self.parameters_list[row][5]
                    idxmax = row
            
            # Compute the mode of the output corresponding to the rule with max firing degree
            Output = mode(self.parameters_list[idxmax][2], keepdims=False).mode
            # Store the output in the preallocated array
            self.y_pred_test[k] = Output
        
        # Check if the original y were string
        if self.reverse_mapping is not None:
            self.y_pred_test = [self.reverse_mapping.get(val) for val in self.y_pred_test]
            
        return self.y_pred_test
    
    # Mapping function for string to numeric
    def map_str_to_numeric(self, y):
        unique_values = np.unique(y)
        mapping = {val: idx for idx, val in enumerate(unique_values)}
        mapped_values = [mapping[val] for val in y]
        
        return mapped_values, mapping
    
    def tau(self, x):
        
        # Variable to sum tau
        sum_tau = 0
        for row in range(len(self.parameters_list)):
            if self.fuzzy_operator == "prod":
                tau = np.prod(self.Gaussian_membership(
                    self.parameters_list[row][0], x,  self.parameters_list[row][1]))
            elif self.fuzzy_operator == "max":
                tau = np.max(self.Gaussian_membership(
                    self.parameters_list[row][0], x,  self.parameters_list[row][1]))
            elif self.fuzzy_operator == "min":
                tau = np.min(self.Gaussian_membership(
                    self.parameters_list[row][0], x,  self.parameters_list[row][1]))
            elif self.fuzzy_operator == "minmax":
                tau = (np.min(self.Gaussian_membership(
                    self.parameters_list[row][0], x,  self.parameters_list[row][1]))
                    * np.max(self.Gaussian_membership(
                    self.parameters_list[row][0], x,  self.parameters_list[row][1])))
            elif self.fuzzy_operator == "equal":
                tau = 1
            
            # Check if it is necessary to multiply tau by the number of observations
            if self.ponder == True:
                tau *= self.parameters_list[row][3]
                
            self.parameters_list[row][4] = max(tau, 1e-10)  # Avoid zero values
            sum_tau += max(tau, 1e-10)
        
        return sum_tau

    def firing_degree(self, x):
        
        # Initialize the total sum
        sum_tau = self.tau(x)
        
        if sum_tau == 0:
            sum_tau = 1 / self.parameters.shape[0]
        for row in range(len(self.parameters_list)):
            self.parameters_list[row][5] = self.parameters_list[row][4] / sum_tau