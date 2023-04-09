# The FeatureSelector class is a tool for selecting important features from a dataset using a given
# model and a specified importance threshold.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    r2_score,
)
from sklearn.base import ClassifierMixin, RegressorMixin


class FeatureSelector:
    """
    Class for selecting important features from a dataset using a given model.

    Attributes:
        threshold (float): Importance threshold for features.
        model (object): Model object for selecting features.
        target_col (str): Target column name in the dataset.
        n_splits (int): Number of splits to be used in StratifiedKFold.
        important_features (list): List of important feature names.
        importance_method (str): Method for calculating feature importances (Shap or Built-in).
        feature_importances (numpy.ndarray): Array of feature importances.
        features (list): List of all feature names.
        ignore_cols (list): List of column names to be ignored in the feature selection.

    Methods:
        fit: Fit the feature selector to the given dataset.
        transform: Transform the given dataset to keep only important features.
        fit_transform: Fit the feature selector to the given dataset and transform it to keep only important features.
        plot_top_n_feats: Plot the top n most important features and their cumulative sum.

    """
    
    def __init__(
        self,
        model,
        target_col,
        n_splits,
        threshold,
        importance_method,
        ignore_cols=None,
    ):
        """
        Initialize the FeatureSelector class.

        Args:
            model (object): Model object for selecting features.
            target_col (str): Target column name in the dataset.
            n_splits (int): Number of splits to be used in StratifiedKFold.
            threshold (float): Importance threshold for features.
            importance_method (str): Method for calculating feature importances.
            ignore_cols (list): List of column names to be ignored in the feature selection.

        Returns:
            None.
            
        """
        self.threshold = threshold
        self.model = model
        self.target_col = target_col
        self.n_splits = n_splits
        self.important_features = None
        self.importance_method = importance_method
        self.feature_importances = None
        self.features = []
        self.ignore_cols = ignore_cols

    def fit(self, df):
        """
        Fit the feature selector to the given dataset.

        Args:
            df (pandas.DataFrame): DataFrame containing the dataset.

        Returns:
            None.

        """
        drop_cols = [self.target_col] + self.ignore_cols
        X = df.drop(drop_cols, axis=1)
        y = df[self.target_col]
        roc_auc = 0
        pr_auc = 0
        rmse = 0
        r2 = 0

        self.feature_importances = np.zeros(X.shape[1])
        self.features = X.columns.to_list()

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2023)

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"Fold {i + 1}")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            self.model.fit(X_train, y_train)
            model_type = self.detect_model_type(self.model)

            if self.importance_method == "shap" and model_type == "classifier":
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X)
                shap_values_df = pd.DataFrame(shap_values[1])
                shap_df = pd.DataFrame(
                    shap_values_df.abs().mean(axis=0).values,
                    columns=["shap"],
                    index=X.columns,
                )
                self.feature_importances += shap_df["shap"].to_numpy() / self.n_splits

            elif self.importance_method == "shap" and model_type == "regressor":
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X)
                shap_values_df = pd.DataFrame(shap_values)
                shap_df = pd.DataFrame(
                    shap_values_df.abs().mean(axis=0).values,
                    columns=["shap"],
                    index=X.columns,
                )
                self.feature_importances += shap_df["shap"].to_numpy() / self.n_splits
            else:
                self.feature_importances += (
                    self.model.feature_importances_ / self.n_splits
                )

            if model_type == "classifier":
                roc_auc += (
                    roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
                    / self.n_splits
                )
                pr_auc += (
                    average_precision_score(
                        y_test, self.model.predict_proba(X_test)[:, 1]
                    )
                    / self.n_splits
                )

            if model_type == "regressor":
                rmse += (
                    np.sqrt(mean_squared_error(y_test, self.model.predict(X_test)))
                    / self.n_splits
                )
                r2 += r2_score(y_test, self.model.predict(X_test)) / self.n_splits

        if model_type == "classifier":
            print("CV Scores of Initial Model:")
            print(f"ROC-AUC: {roc_auc}, PR_AUC: {pr_auc}")
        else:
            print("CV Scores of Initial Model:")
            print(f"RMSE: {rmse}, R2: {r2}")

        self.important_features = [
            feature
            for feature, importance in zip(X.columns, self.feature_importances)
            if importance > self.threshold
        ]

    def transform(self, df):
        """
        Transforms the given dataset to keep only important features.

        Args:
            df (pandas.DataFrame): DataFrame containing the dataset.

        Returns:
            pandas.DataFrame: DataFrame containing only the important features.

        """
        df_important = df[
            self.important_features + [self.target_col] + self.ignore_cols
        ]
        print(
            f"{len(self.features)-len(self.important_features)} columns dropped due to 0 significance"
        )
        return df_important

    def fit_transform(self, df):
        """
        Fits the feature selector to the given dataset and transforms it to keep only important features.

        Args:
            df (pandas.DataFrame): DataFrame containing the dataset.

        Returns:
            pandas.DataFrame: DataFrame containing only the important features.

        """
        self.fit(df)
        return self.transform(df)

    def plot_top_n_feats(self, top_n_feats):
        """
        Plots the top n most important features and their cumulative sum.

        Args:
            n (int): Number of top features to plot.

        Returns:
            None.

        """
        # Normalize feature importances to understand partial effect of each feature among all features.
        fimps = pd.Series(
            data=self.feature_importances, index=self.features
        ).sort_values(ascending=False)
        fimps /= fimps.sum()

        fig, ax1 = plt.subplots(figsize=(20, 12))

        sns.barplot(x=fimps.index[:top_n_feats], y=fimps[:top_n_feats], ax=ax1)
        ax1.set_ylabel("Importance")
        plt.xticks(rotation=90)
        ax1.set_title(f"Top {top_n_feats} Feature Importances")
        ax1.set_xlabel("Feature")

        ax2 = ax1.twinx()
        ax2.plot(
            fimps.index[:top_n_feats],
            fimps.cumsum()[:top_n_feats],
            label="Cumulative-Sum",
        )
        ax2.legend()
        ax2.set_ylabel("Cumulative-Sum")

    def detect_model_type(self, model):
        """
        This function takes a model object as input and returns a string indicating whether
        the model is a classifier or a regressor.

        Args:
            model (object): Model object.

        Returns:
            str: Type of the given model.

        """
        if isinstance(model, ClassifierMixin):
            return "classifier"
        elif isinstance(model, RegressorMixin):
            return "regressor"
        else:
            raise ValueError("The given model is not a valid classifier or regressor.")