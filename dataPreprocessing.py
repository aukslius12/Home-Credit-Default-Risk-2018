# Dependencies.
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import fancyimpute

class dataPreprocessing:
    '''
    Class for handling all of the dataPreprocessing, with customizable parameters. 
    
    Use dataPreprocessing.preprocess()
    
    Attributes
    ----------
    non_categorical_ints: list of strings,
    
    fill_na_value: str, to fill NA values in categorical variables.
    
    remove_na_values: boolean, used for non-parametric models which handle missing values themselves.
    
    parametric_model: boolean, used to indicate whether data will be modeled with a parametric model or not.
    
    dataset_is_kaggle_comp: boolean, this class is used for https://www.kaggle.com/c/home-credit-default-risk/data#_=_ this specific dataset. Set to False for any other dataset.
        
    '''
    def __init__(self,
                 non_categorical_ints = ['DAYS_BIRTH', 'DAYS_ID_PUBLISH', 'CNT_CHILDREN', 'DAYS_EMPLOYED'],
                 fill_na_value = 'not specified',
                 remove_na_values = False,
                 parametric_model = True,
                 dataset_is_kaggle_comp = True):
        
        self.non_categorical_ints = non_categorical_ints
        self.fill_na = fill_na_value
        self.remove_na = remove_na_values
        self.is_param = parametric_model
        self.kaggle = dataset_is_kaggle_comp
        
        
    def preprocess(self, df, inplace = False):
        '''
        Handles all of the preprocessing in two steps:
            * Processes the categorical part (dataPreprocessing.categorical_preprocessing())
            * Processes the continuos part (dataPreprocessing.cont_parametric_preprocessing() or cont_nonparametric_preprocessing())
        
        Merges everything into a single dataframe and returns a pandas.DataFrame object.
        
        Attributes
        ----------
        inplace: boolean, set to False if you want your dataframe returned. If true, works like fit_transform in sklearn methods.
        
        All further attributes are defined in the __init__() function.
        
        '''        
        self.categorical_preprocessing(df)
        self.continuous_fork(df)
        self.df = pd.concat([self.df_continuous, self.df_categorical], axis=1)
        if not inplace:
            return self.df
    
        
    def continuous_fork(self, df):
        '''
        Switches between data preprocessing for:
            * Parametric models, where NA continuous values have meaning (for example, NA = no car);
            * Non-parametric models, where NA is specified outside of the value scope (for example, -1);
            * Non-parametric models, which deal with NA's their own way.
            
        Attributes
        ----------
        df: pandas.DataFrame, of any type.
        
        '''
        if self.is_param:
            self.cont_parametric_preprocessing(df)
        else:
            self.cont_nonparametric_preprocessing(df)
            
    
    def cont_parametric_preprocessing(self, df, inplace = True):
        '''
        Implementing mice to fill missing values and to add a column of indicators for whether that value was missing or not. 
                
        Used for parametric models, where NA continuous values have meaning (for example, NA = no car).
        
        Attributes
        ----------
        df: pandas.DataFrame, of any type.
        
        inplace: boolean, set to False if you want your dataframe returned.
        
        '''
        def parametric_input(df):
            '''
            Using mice to fill missing continuous variables, adding a column to indicate that they were missing.
            
            Returns a transformed dataframe.
            
            Attributes
            ----------
            df: pandas.DataFrame, type of float or int types.
            
            '''
            # Filling values using mice.
            mice_matrix = fancyimpute.MICE(n_imputations=50).complete(df.values)
            mice_df = pd.DataFrame(mice_matrix)
            mice_df.columns = df.columns
            mice_df.index = df.index

            # Adding an indicator dataframe.
            ismissing_df = create_ismissing_df(df)
            return(pd.concat([mice_df, ismissing_df], axis=1))
        

        def create_ismissing_df(df):
            '''
            Creates an additional dataframe which indicates which rows in previous data frame had missing data
            in their variables.
            
            Returns a pandas dataframe.
            
            Attributes
            ----------
            df: pandas.DataFrame, type of float or int types.
            
            '''
            df_new = pd.DataFrame()
            for col in df.columns:
                missing = df[col].isnull()
                df_new = pd.concat([df_new, pd.DataFrame({col + '_IS_NA': missing})], axis=1)
            return(df_new)
        
        
        # THIS IS ONLY USED FOR A SPECIFIC COMPETITION.
        if self.kaggle:
            # Transforming HOURS_ into sin((2*pi*HOURS_/24)) to imitate daily loan application ammount fluctuations.
            # Another consideration would be to use it as categorical variables.
            df['HOUR_APPR_PROCESS_START'] = np.sin((2*np.pi*df['HOUR_APPR_PROCESS_START'].values)/24)

            # Source: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
            # Replace the anomalous values with nan
            df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})

        # DAYS_EMPLOYED becomes floa64 after adding nan's. [:1] to avoid duplication.
        df = pd.concat([df.select_dtypes('float64'), df[self.non_categorical_ints[:3]]], axis=1, sort=False)
        # Uses mice to fill the values.
        df = parametric_input(df)

        # Testing for any missing values still left.
        if any(df.apply(pd.Series.isna).apply(any)):
            Warning('Continuous parametric preprocessing has failed. Missing values still present.')
        
        # Finalizing the transformed categorical part of the dataframe.
        self.df_continuous = df
        if not inplace:
            return(self.df_continuous)
        
        
    def cont_nonparametric_preprocessing(self, df, inplace = True):
        '''
        Replaces NA's with an arbitrary value or leaves NA's where they are for models that can deal with it.
        
        Used for non-parametric models, where NA is specified outside of the value scope (for example, -1)
        
        Attributes
        ----------
        df: pandas.DataFrame, of numeric values.
        
        inplace: boolean, set to False if you want your dataframe returned.
        
        '''
        df = df.copy(deep=True) # Fixes a bug when initializing class multiple times. Too lazy to rewrite everything, lesson already learned :).
        
        def repl_outside_scope(series):
            '''
            Replaces missing values with arbitrary values, for example with -1 when data is CAR_AGE.

            Attributes
            ----------
            series: pandas.Series, float or integer only.

            '''            
            if not series.isna().any():
                return series

            # Get non-na/null values.
            not_missing = series[~series.isna() | ~series.isnull()]

            ## Rules for creating the arbitrary value, outside the scope
            # All negative or all positive values.
            if all(not_missing <= 0):
                return series.fillna(1)
            elif all(not_missing >=0):
                return series.fillna(-1)
            # If distribution intersects y-axis.
            else:
                lo, hi = not_missing.min(), not_missing.max()
                if abs(lo) < abs(hi):
                    # 3 standard deviations outside the lowest value, if data is skewed left.
                    return series.fillna(lo*3*not_missing.std())
                elif abs(lo) > abs(hi):
                    # 3 standard deviations outside the highest value if data is skewed right.
                    return series.fillna(hi*3*not_missing.std())
                else:
                    return series.fillna(10**10) # If all else fails.
                

        # THIS IS ONLY USED FOR A SPECIFIC COMPETITION.
        if self.kaggle:
            # Transforming HOURS_ into sin((2*pi*HOURS_/24)) to imitate daily loan application ammount fluctuations.
            # Another consideration would be to use it as categorical variables.
            df['HOUR_APPR_PROCESS_START'] = np.sin((2*np.pi*df['HOUR_APPR_PROCESS_START'].values)/24)

            # Source: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
            # Replace the anomalous values with nan
            df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})

        # DAYS_EMPLOYED becomes floa64 after adding nan's. [:3] to avoid duplication.
        df = pd.concat([df.select_dtypes('float64'), df[self.non_categorical_ints[:3]]], axis=1, sort=False)
        
        # Missing value handling.
        if self.remove_na:
            self.df_continuous = df.apply(repl_outside_scope)
        else:
            # Returning with NA's present.
            self.df_continuous = df
        
        if not inplace:
            return(self.df_continuous)
        
        
    def categorical_preprocessing(self, df = pd.DataFrame(), fill_na_value = '', inplace=True):
        '''
        Encodes and creates a sparse matrix out of categorical variables.
        Fills missing values with 'not specified'.
        
        Attributes
        ----------
        df: pandas.DataFrame, of any type.
        
        fill_na_value: str, to fill NA values.
        
        inplace: boolean, set to False if you want your dataframe returned.
        
        '''
        # Initializing default values.
        fill_na_value = self.fill_na
        
        def replace_xna(df, replace_with = fill_na_value):
            '''
            Finds and replaces XNA values if any.
            
            Attributes
            ----------
            df: pandas.DataFrame of str.
            
            replace_with: str to replace 'XNA' with. 
            
            '''
            # Get colnames of columns with XNA in them.
            xna_cols = df.apply(lambda col: any(col == 'XNA'))
            colnames = xna_cols[xna_cols == True].index.format()
            
            if len(colnames) != 0:                
                df[colnames] = df[colnames].replace('XNA', replace_with)
            
            return df
        
        
        def encode_transform(series):
            '''
            Encodes and trasforms categorical series.
            
            Returns sparse matrix of encoded variables - 1 column to avoid dummy variable trap. Also names variables according to their true value for better further variable importance interpretation.
            
            Attributes
            ----------
            series: pandas.Series type of any values. Can also include numeric 0, 1, 2,.. values.
            
            '''
            series_transf = LabelEncoder().fit_transform(series)
            
            # If more than 2 categories, dummy variables are needed.
            if len(series.unique()) > 1:
                dummies = pd.get_dummies(series)
                dummies.columns = [str(val) for val in dummies.columns.values] # Fixes numeric value problems.
                dummies.columns = series.name + '_' + dummies.columns
                series_transf = dummies.iloc[:, 1:] # Avoiding the dummy variable trap.
            else:
                # 1 indicates the 2nd unique value, as 0 is always chosen as the first value.
                series_transf = pd.Series(series_transf, name=series.name + '_' + series.unique()[1])
            return series_transf
        
        # Source: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
        # Create an anomalous flag column
        df['DAYS_EMPLOYED_ANOM'] = df['DAYS_EMPLOYED'] == 365243

        # Removing float64 types and non categorical ints.
        df = df.select_dtypes(exclude='float64').drop(self.non_categorical_ints, axis=1)
        
        # Manually replacing missing categorical variables.
        df = df.fillna(fill_na_value)

        # Checks if any NA's are left.
        if any(df.apply(pd.Series.isna).apply(any)):
            Warning('Categorical preprocessing has failed. Missing values still present.')
        
        # Replacing XNA's.
        df = replace_xna(df)
        
        # Encoding and transforming the variables.
        df_final = pd.DataFrame()
        for col in df.columns:
            df_final = pd.concat([df_final, encode_transform(df[col])], axis=1)
            
        # Finalizing the transformed categorical part of the dataframe.
        self.df_categorical = df_final
        if not inplace:
            return(df_final)
        
    