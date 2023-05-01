import pandas as pd
from category_encoders import TargetEncoder, CatBoostEncoder, OrdinalEncoder
from category_encoders.wrapper import NestedCVWrapper
from datetime import datetime
import os
from tqdm import tqdm
import time
import pickle

class Encoder:
    """
This module provides an encoder class for categorical variable encoding using target, catboost, and ordinal encoding methods. 
For columns with high cardinality, i.e., columns with a large number of unique values, the class uses a technique called 
frequency-based binning to identify low-frequency categories. This technique identifies categories that occur less frequently
than a specified minimum frequency threshold and replaces them with a new category called LowFrequencyCategory.The high_cardinality_threshold
parameter in the Encoder class controls the threshold value above which a column is considered to have high cardinality. If a column's unique
value count is greater than this threshold, the Encoder class applies frequency-based binning to it. This method helps reduce the number of unique
categories in high-cardinality columns and helps to deal with the "curse of dimensionality" problem, which can arise when working with large datasets
that have many categorical features.

Attributes:
    min_freq (float): Minimum frequency for low-frequency category binning. Default is 0.05.
    cv (int): Number of cross-validation folds for target and catboost encoding. Default is 5.
    ordinal_threshold (int): Threshold for number of unique values in a categorical variable to be considered as an ordinal variable. Default is 3.
    high_cardinality_threshold (int): Threshold for number of unique values in a categorical variable to be considered as high cardinality. Default is 30.
    columns (list): List of column names to apply encoding to. If None, all object dtype columns will be encoded. Default is None.
    ignore_cols (list): List of column names to exclude from encoding. Default is None.
    target_col (str): Name of target column. Default is 'Target'.
    export (bool): Whether to export fitted encoders as pickle files. Default is True.

Methods:
    fit(df): Fits the encoders using the provided DataFrame.
    transform(df): Encodes categorical variables of the provided DataFrame using the fitted encoders.
    fit_transform(df): Fits the encoders using the provided DataFrame and encodes categorical variables.

Examples:
    To use the encoder on a DataFrame:

    >>> encoder = Encoder()
    >>> encoded_df = encoder.fit_transform(df)

"""

    def __init__(self, min_freq=0.05, cv=5, ordinal_threshold = 3, high_cardinality_threshold=30, columns=None, ignore_cols=None, target_col='Target', export=True):
        self.min_freq = min_freq
        self.cv = cv
        self.ordinal_threshold = ordinal_threshold
        self.high_cardinality_threshold = high_cardinality_threshold
        self.columns = columns
        self.ignore_cols = ignore_cols
        self.target_col = target_col
        self.export = export
        
    def fit(self, df):
        if self.columns:
            self.bin_cols = [col for col in self.columns if col not in self.ignore_cols and df[col].dtype == 'object']
        else:
            self.bin_cols = [col for col in df.columns if col not in self.ignore_cols and df[col].dtype == 'object']
        
        self.ordinal_cols = []
        self.target_cols = []
        self.catboost_cols = []

        print('Categorical columns number:', len(self.bin_cols))
        print('----------------------------------')

        for col in tqdm(self.bin_cols, desc='Columns'):
            unique_count = len(df[col].unique())
            if unique_count <= self.ordinal_threshold:
                self.ordinal_cols.append(col)
            elif unique_count > self.high_cardinality_threshold:
                freq = df[col].value_counts(normalize=True)
                mask = freq > self.min_freq
                mask = mask[mask].index.tolist()
                if len(mask) < len(freq):
                    df[col] = df[col].replace(freq[list(set(freq.index)-set(mask))].index, 'LowFrequencyCategory')
                    if len(df[col].unique()) <= self.ordinal_threshold:
                        self.ordinal_cols.append(col)
                    else:
                        self.target_cols.append(col)
                        self.catboost_cols.append(col)
                else:
                    self.target_cols.append(col)
                    self.catboost_cols.append(col)
            else:
                self.target_cols.append(col)
                self.catboost_cols.append(col)
        print('Low Frequency Category Binning and Encoding Method Separation Completed')
        print('----------------------------------')
        
        print('Number of Ordinal Columns:',len(self.ordinal_cols))
        start = time.time()
        #self.ordinal_encoder = pd.get_dummies(df[self.ordinal_cols], prefix_sep='_', drop_first=True)
        self.ordinal_encoder = OrdinalEncoder(cols=self.ordinal_cols)
        self.ordinal_encoder.fit(df.loc[:,self.ordinal_cols])
        end = time.time()
        print('Ordinal Encoding took:', end - start)
        print('----------------------------------')

        print('Number of Target Encode Columns:',len(self.target_cols))
        start = time.time()
        self.target_encoder = NestedCVWrapper(TargetEncoder(cols=self.target_cols), cv=self.cv, random_state=42)
        self.target_encoder.fit(df.loc[:,self.target_cols], df[self.target_col].astype(int))
        end = time.time()
        print('Target Encoding took:', end - start)
        print('----------------------------------')

        print('Number of Catboost Encode Columns:',len(self.catboost_cols))
        start = time.time()
        self.catboost_encoder = NestedCVWrapper(CatBoostEncoder(cols=self.catboost_cols), cv=self.cv, random_state=42)
        self.catboost_encoder.fit(df.loc[:,self.catboost_cols], df[self.target_col].astype(int))
        end = time.time()
        print('Catboost Encoding took:', end - start)
        print('----------------------------------')
        
        if self.export:
            
            print("Exporting Encoders...", "\n")
            
            now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            
            
            if os.path.exists("./encoders") == False:
                
                os.mkdir("./encoders")
                
                os.mkdir(f"./encoders/{now}")
                
            else:
                
                os.mkdir(f"./encoders/{now}")
                
            with open(f"./encoders/{now}/ordinal_encoder.pkl",'wb') as f:
                pickle.dump(self.ordinal_encoder,f)

            with open(f"./encoders/{now}/target_encoder.pkl",'wb') as f:
                pickle.dump(self.target_encoder,f)

            with open(f"./encoders/{now}/catboost_encoder.pkl",'wb') as f:
                pickle.dump(self.catboost_encoder,f)
                    
            print("Exporting Completed!", "\n")
            
        return self
    
    def transform(self, df):
        df_o = df[self.ordinal_cols].copy()
        df_o = self.ordinal_encoder.transform(df_o)
        df_o = df_o.add_prefix('ordinal_')
        df = pd.concat([df, df_o], axis=1)

        df_t = df[self.target_cols].copy()
        df_t = self.target_encoder.transform(df_t)
        df_t = df_t.add_prefix('target_')
        df = pd.concat([df, df_t], axis=1)

        df_c = df[self.catboost_cols].copy()
        df_c = self.catboost_encoder.transform(df_c)
        df_c = df_c.add_prefix('catboost_')
        df = pd.concat([df, df_c], axis=1)
        df = df.drop(columns=self.bin_cols)
        
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)