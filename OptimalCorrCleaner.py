import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def opt_corr_cleaner(df, features_sorted_by_importance, corr_threshold=0.7):
    """
    Removes high correlation pairs that have low importance from a pandas DataFrame.
    
    Parameters:
        - df: pandas DataFrame containing the features
        - features_sorted_by_importance: list of feature names sorted by importance
        - corr_threshold: correlation threshold (default: 0.7)
    
    Returns:
        - tuple containing:
            - list of remaining feature names after correlation elimination
            - list of dropped feature names
    
    Description:
        This function removes high correlation pairs from a pandas DataFrame that have low importance. The function takes as input a DataFrame,
        a list of feature names sorted by importance, and a correlation threshold. The function calculates the correlation matrix for the input
        DataFrame and plots it using seaborn. The function then finds the high correlation pairs that exceed the correlation threshold and 
        sorts them in descending order of correlation strength. For each high correlation pair, the function removes the feature with the lower
        importance from the list of remaining features and adds it to the list of dropped features. If a feature has already been dropped in a 
        previous iteration, the function removes that feature and its pairs from the high_corr_pairs list so that it is not considered again in 
        subsequent iterations. In this way, undesired feature elimination is prevented. The function returns a tuple containing the list of remaining feature names after correlation elimination and 
        the list of dropped feature names.
    """
    
    # Calculate the correlation matrix
    corr_matrix = df[features_sorted_by_importance].corr()
    
    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f', square=True)
    plt.title('Correlation Matrix')
    plt.show()
    
    # Find the high correlation pairs
    high_corr_pairs = np.where(np.abs(corr_matrix) >= corr_threshold)
    high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*high_corr_pairs) if x < y]
    high_corr_pairs.sort(key=lambda x: abs(corr_matrix.loc[x[0], x[1]]), reverse=True)
    
    # Remove high correlation pairs
    dropped_features = []
    remaining_features = features_sorted_by_importance.copy()
    for pair in high_corr_pairs:
        feature1, feature2 = pair
        if feature1 in remaining_features and feature2 in remaining_features:
            importance1 = remaining_features.index(feature1)
            importance2 = remaining_features.index(feature2)
            if importance1 > importance2:
                remaining_features.remove(feature1)
                dropped_features.append(feature1)
                # Remove feature1 from high_corr_pairs
                high_corr_pairs = [p for p in high_corr_pairs if p[0] != feature1 and p[1] != feature1]
            elif importance1 < importance2:
                remaining_features.remove(feature2)
                dropped_features.append(feature2)
                # Remove feature2 from high_corr_pairs
                high_corr_pairs = [p for p in high_corr_pairs if p[0] != feature2 and p[1] != feature2]
            else:
                remaining_features.remove(feature1)
                dropped_features.append(feature1)
                # Remove feature1 from high_corr_pairs
                high_corr_pairs = [p for p in high_corr_pairs if p[0] != feature1 and p[1] != feature1]
    
    return remaining_features, dropped_features
