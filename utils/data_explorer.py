import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.feature_selection import mutual_info_regression
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import warnings
from colorama import Fore, Style

# Constants
DEFAULT_VISUALIZATION_SETTINGS = {
    'color_palette': 'viridis',
    'figure_size': (10, 8),
    'font_size': 12,
    'title_font_size': 16
}

STATISTICAL_TEST_REGISTRY = {
    'shapiro-wilk': {
        'function': stats.shapiro,
        'params': {},
        'description': 'Shapiro-Wilk test for normality'
    },
    'd_agostino_k2': {
        'function': stats.normaltest,
        'params': {},
        'description': "D'Agostino's K^2 test for normality"
    }
}

CORRELATION_METHODS = {
    'pearson': {
        'description': 'Pearson correlation coefficient measures the linear relationship between two datasets.',
        'assumptions': 'Assumes data is normally distributed.',
        'interpretation': 'Values range from -1 to 1, where 0 implies no correlation.'
    },
    'spearman': {
        'description': 'Spearman rank correlation calculates the monotonic relationship between two datasets.',
        'assumptions': 'Does not assume data is normally distributed.',
        'interpretation': 'Values range from -1 to 1, assessing how well the relationship between two variables can be described using a monotonic function.'
    },
    'kendall': {
        'description': "Kendall's Tau measures the correspondence between two rankings.",
        'assumptions': 'Less sensitive to outliers compared to Pearson.',
        'interpretation': 'Values range from -1 to 1, with higher values indicating stronger agreement in rankings.'
    }
}

OUTLIER_DETECTION_METHODS = {
    'iqr': {
        'description': 'IQR method identifies outliers based on the interquartile range.',
        'params': {'threshold': 1.5},
        'interpretation': 'Values outside Q1-threshold*IQR and Q3+threshold*IQR are flagged as outliers.'
    },
    'z-score': {
        'description': 'Z-score method identifies outliers based on standard deviations from the mean.',
        'params': {'threshold': 3},
        'interpretation': 'Observations beyond the specified threshold standard deviations are identified.'
    }
}


def normalize_feature_names(df):
    """
    Standardizes feature names by converting them to lowercase, replacing spaces with underscores,
    and removing special characters.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with normalized column names.
    """
    df.columns = [
        ''.join(c if c.isalnum() else '_' for c in str(x))
        .lower()
        .replace('__', '_') for x in df.columns
    ]
    return df


def detect_data_types(df, numeric_threshold=0.7):
    """
    Intelligently identifies the appropriate data type for each column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_threshold (float): Threshold to determine if a column should be numeric.

    Returns:
        tuple: Lists of numeric and categorical columns.
    """
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            # Check if the column contains mostly numeric values
            try:
                numeric_count = pd.to_numeric(df[col], errors='coerce').notnull().sum()
                if numeric_count / len(df) >= numeric_threshold:
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            except:
                categorical_cols.append(col)
    return numeric_cols, categorical_cols


def create_feature_scatter_matrix(df, features, target=None, samples=1000):
    """
    Generates a scatter plot matrix for a subset of important features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): List of features to plot.
        target (str, optional): Target variable for coloring. Defaults to None.
        samples (int, optional): Number of samples to use for plotting. Defaults to 1000.

    Returns:
        plt.figure: Scatter matrix plot.
    """
    if len(df) > samples:
        df = df.sample(n=samples, random_state=42)  # for reproducibility

    if target:
        fig = sns.pairplot(df[features + [target]], hue=target)
    else:
        fig = sns.pairplot(df[features])
    return fig


class DataExplorer:
    """
    Centralizes functionality for thorough data exploration and statistical analysis of health indicators.
    """

    def __init__(self, features_df, target_df, output_dir='reports', config=None):
        """
        Initializes the DataExplorer with feature and target data.

        Args:
            features_df (pd.DataFrame): DataFrame containing the features.
            target_df (pd.Series): Series containing the target variable.
            output_dir (str, optional): Directory to save reports. Defaults to 'reports'.
            config (dict, optional): Configuration parameters. Defaults to None.
        """
        self.features_df = features_df
        self.target_df = target_df
        self.combined_df = pd.concat([features_df, target_df], axis=1)
        self.exploration_results = {}
        self.logger = self._setup_logger()
        self.numeric_features, self.categorical_features = detect_data_types(features_df)
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.config = config if config is not None else {}

    def _setup_logger(self):
        """
        Sets up a basic logger for the class.

        Returns:
            logging.Logger: Logger object.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # Create a handler that writes log messages to the console
        handler = logging.StreamHandler()
        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_data(self, features_path, target_path):
        """
        Loads preprocessed feature and target datasets from specified file paths.

        Args:
            features_path (str): Path to the features CSV file.
            target_path (str): Path to the target CSV file.

        Returns:
            pd.DataFrame: The combined DataFrame for further analysis.
        """
        try:
            self.features_df = pd.read_csv(features_path)
            self.target_df = pd.read_csv(target_path)
            
            # Basic validation checks
            if not isinstance(self.features_df, pd.DataFrame) or not isinstance(self.target_df, pd.DataFrame):
                raise ValueError("Features and target must be pandas DataFrames.")
            
            if self.features_df.empty or self.target_df.empty:
                raise ValueError("Features or target DataFrame is empty.")
            
            # Assuming both DataFrames have a common index column, e.g., 'district_id'
            if 'district_id' not in self.features_df.columns or 'district_id' not in self.target_df.columns:
                raise ValueError("Both features and target DataFrames must have a common index column 'district_id'.")
            
            # Merge the DataFrames on the common index
            self.combined_df = pd.merge(self.features_df, self.target_df, on='district_id', how='inner')
            
            if self.combined_df.empty:
                raise ValueError("The combined DataFrame is empty after merging. Check the common index.")
            
            self.logger.info("Data loaded and combined successfully.")
            return self.combined_df
        
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"Data validation error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            raise

    def get_basic_stats(self, include_percentiles=True, include_shape_metrics=True):
        """
        Calculates comprehensive descriptive statistics for all numeric features in the dataset.

        Args:
            include_percentiles (bool, optional): Whether to include percentiles. Defaults to True.
            include_shape_metrics (bool, optional): Whether to include shape metrics. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame of statistics organized by feature.
        """
        stats_dict = {}
        for col in self.numeric_features:
            stats = {}
            stats['count'] = self.features_df[col].count()
            stats['mean'] = self.features_df[col].mean()
            stats['median'] = self.features_df[col].median()
            stats['std'] = self.features_df[col].std()
            stats['min'] = self.features_df[col].min()
            stats['max'] = self.features_df[col].max()

            if include_percentiles:
                stats['5th'] = self.features_df[col].quantile(0.05)
                stats['25th'] = self.features_df[col].quantile(0.25)
                stats['75th'] = self.features_df[col].quantile(0.75)
                stats['95th'] = self.features_df[col].quantile(0.95)

            if include_shape_metrics:
                stats['skewness'] = self.features_df[col].skew()
                stats['kurtosis'] = self.features_df[col].kurtosis()

            stats_dict[col] = stats

        stats_df = pd.DataFrame(stats_dict).T
        self.exploration_results['basic_stats'] = stats_df
        self.logger.info("Basic statistics calculated.")
        return stats_df

    def analyze_correlations(self, threshold=0.7, method='pearson', visualize=True, figsize=(12, 10)):
        """
        Identifies and visualizes feature correlations.

        Args:
            threshold (float, optional): Correlation threshold. Defaults to 0.7.
            method (str, optional): Correlation method. Defaults to 'pearson'.
            visualize (bool, optional): Whether to visualize the correlation matrix. Defaults to True.
            figsize (tuple, optional): Figure size for the heatmap. Defaults to (12, 10).

        Returns:
            tuple: Correlation matrix and list of highly correlated feature pairs.
        """
        try:
            # Calculate the correlation matrix
            corr_matrix = self.features_df[self.numeric_features].corr(method=method)
            
            # Identify highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) >= threshold:
                        col_i = corr_matrix.columns[i]
                        col_j = corr_matrix.columns[j]
                        high_corr_pairs.append((col_i, col_j, corr_matrix.iloc[i, j]))
            
            # Visualize the correlation matrix
            if visualize:
                plt.figure(figsize=figsize)
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
                plt.title(f"Correlation Matrix ({method.capitalize()} Method)")
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"correlation_matrix_{method}.png"))
                plt.close()
                self.logger.info(f"Correlation matrix visualized and saved to {self.output_dir}")
            
            self.exploration_results['correlations'] = {'matrix': corr_matrix, 'high_corr_pairs': high_corr_pairs}
            self.logger.info("Correlation analysis completed.")
            return corr_matrix, high_corr_pairs
        
        except Exception as e:
            self.logger.error(f"Error during correlation analysis: {e}")
            raise

    def identify_outliers(self, method='iqr', threshold=1.5, visualize=True):
        """
        Detects outliers across all numeric features using the selected method.

        Args:
            method (str, optional): Outlier detection method. Defaults to 'iqr'.
            threshold (float, optional): Threshold for outlier detection. Defaults to 1.5.
            visualize (bool, optional): Whether to visualize outliers. Defaults to True.

        Returns:
            dict: Mapping feature names to arrays of outlier indices, along with summary statistics.
        """
        outlier_indices = {}
        outlier_stats = {}

        for col in self.numeric_features:
            data = self.features_df[col]
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = data[(data < lower_bound) | (data > upper_bound)].index
            elif method == 'z-score':
                z_scores = np.abs(stats.zscore(data))
                outliers = data[z_scores > threshold].index
            else:
                raise ValueError(f"Unsupported outlier detection method: {method}")

            outlier_indices[col] = outliers.tolist()
            outlier_stats[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(data) * 100
            }

            if visualize:
                plt.figure(figsize=(8, 6))
                if method == 'iqr':
                    sns.boxplot(x=data)
                else:
                    sns.histplot(data, kde=True)
                plt.title(f'Outliers in {col} ({method} Method)')
                plt.savefig(os.path.join(self.output_dir, f'outliers_{col}_{method}.png'))
                plt.close()
                self.logger.info(f"Outliers in {col} visualized and saved.")

        self.exploration_results['outliers'] = {'indices': outlier_indices, 'stats': outlier_stats}
        self.logger.info("Outlier analysis completed.")
        return outlier_indices, outlier_stats

    def perform_feature_importance(self, method='mutual_info', target=None, n_neighbors=3, discrete_features='auto'):
        """
        Estimates initial feature importance using non-model specific methods.

        Args:
            method (str, optional): Method for feature importance. Defaults to 'mutual_info'.
            target (str, optional): Target variable. Defaults to None.
            n_neighbors (int, optional): Number of neighbors for mutual information. Defaults to 3.
            discrete_features (str, optional): Whether features are discrete. Defaults to 'auto'.

        Returns:
            pd.DataFrame: Features ranked by importance score.
        """
        if target is None:
            if 'infant_mortality_rate' in self.combined_df.columns:
                target = 'infant_mortality_rate'
            else:
                raise ValueError("Target variable must be specified.")

        # Create a copy of the features and target for processing
        X = self.features_df[self.numeric_features].copy()
        y = self.combined_df[target].copy()
        
        # Handle missing values before feature importance calculation
        # Option 1: Remove rows with missing values
        # valid_indices = X.dropna().index
        # X = X.loc[valid_indices]
        # y = y.loc[valid_indices]
        
        # Option 2: Impute missing values with mean (faster and preserves all samples)
        for col in X.columns:
            if X[col].isna().any():
                self.logger.info(f"Imputing missing values in {col} for feature importance calculation")
                X[col].fillna(X[col].mean(), inplace=True)
        
        # Check if we still have missing values in the target
        if y.isna().any():
            self.logger.info(f"Removing {y.isna().sum()} rows with missing target values")
            valid_indices = y.dropna().index
            X = X.loc[valid_indices]
            y = y.loc[valid_indices]

        if method == 'mutual_info':
            importance = mutual_info_regression(X, y, n_neighbors=n_neighbors, discrete_features=discrete_features, random_state=42)
            feature_importance = pd.Series(importance, index=X.columns)
            feature_importance = feature_importance.sort_values(ascending=False)
        else:
            raise ValueError(f"Unsupported feature importance method: {method}")

        self.exploration_results['feature_importance'] = feature_importance
        self.logger.info("Feature importance calculated.")
        return feature_importance

    def analyze_missing_values(self, visualize=True, figsize=(12, 6)):
        """
        Performs comprehensive analysis of missing values in the dataset.

        Args:
            visualize (bool, optional): Whether to visualize missing values. Defaults to True.
            figsize (tuple, optional): Figure size for visualizations. Defaults to (12, 6).

        Returns:
            pd.DataFrame: Summarizing missing value statistics for each feature.
        """
        missing_counts = self.features_df.isnull().sum().sort_values(ascending=False)
        missing_percentages = (self.features_df.isnull().sum() / len(self.features_df)).sort_values(ascending=False)
        missing_data = pd.concat([missing_counts, missing_percentages], axis=1, keys=['Count', 'Percentage'])

        if visualize:
            plt.figure(figsize=figsize)
            msno.matrix(self.features_df)
            plt.title('Missing Values Matrix')
            plt.savefig(os.path.join(self.output_dir, 'missing_values_matrix.png'))
            plt.close()

            plt.figure(figsize=figsize)
            msno.bar(self.features_df)
            plt.title('Missing Values Bar Chart')
            plt.savefig(os.path.join(self.output_dir, 'missing_values_bar.png'))
            plt.close()
            self.logger.info("Missing values visualized and saved.")

        self.exploration_results['missing_values'] = missing_data
        self.logger.info("Missing value analysis completed.")
        return missing_data

    def analyze_categorical_features(self, max_categories=20):
        """
        Examines categorical variables in the dataset to understand their cardinality, distribution, and potential encoding needs.

        Args:
            max_categories (int, optional): Maximum number of categories to display. Defaults to 20.

        Returns:
            dict: Categorical feature statistics and recommended preprocessing approaches.
        """
        categorical_stats = {}

        for col in self.categorical_features:
            num_unique = self.features_df[col].nunique()
            value_counts = self.features_df[col].value_counts()
            entropy = stats.entropy(value_counts)

            categorical_stats[col] = {
                'num_unique': num_unique,
                'value_counts': value_counts.to_dict(),
                'entropy': entropy
            }

            plt.figure(figsize=(10, 6))
            if num_unique <= max_categories:
                self.features_df[col].value_counts().plot(kind='bar')
            else:
                self.logger.warning(f"Too many categories in {col} to plot.")
                value_counts[:max_categories].plot(kind='bar')
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(self.output_dir, f'categorical_{col}.png'))
            plt.close()
            self.logger.info(f"Categorical feature {col} analyzed and visualized.")

        self.exploration_results['categorical_features'] = categorical_stats
        self.logger.info("Categorical feature analysis completed.")
        return categorical_stats

    def explore_target_distribution(self, bins=30, kde=True):
        """
        Analyzes the distribution of the target variable.

        Args:
            bins (int, optional): Number of bins for histogram. Defaults to 30.
            kde (bool, optional): Whether to include KDE plot. Defaults to True.

        Returns:
            dict: Target distribution statistics and visualization objects.
        """
        try:
            # Determine the target column name
            if isinstance(self.target_df, pd.DataFrame):
                # If target_df is a DataFrame, get the first column name
                if len(self.target_df.columns) > 0:
                    # Skip the district_id column if present
                    target_cols = [col for col in self.target_df.columns if col != 'district_id']
                    if target_cols:
                        target = target_cols[0]
                    else:
                        target = self.target_df.columns[0]
                else:
                    raise ValueError("Target DataFrame has no columns")
            else:
                # If target_df is a Series, get its name
                target = self.target_df.name
            
            # If we still don't have a target name, use a default
            if target is None:
                target = 'target'
                self.logger.warning("No target name found, using 'target' as default")
            
            # Verify the target exists in the combined DataFrame
            if target not in self.combined_df.columns:
                raise ValueError(f"Target column '{target}' not found in the combined DataFrame")
                
            # Check if we have sufficient data for visualization
            target_data = pd.to_numeric(self.combined_df[target], errors='coerce').dropna()
            if len(target_data) < 2:
                self.logger.warning("Insufficient data for target distribution analysis")
                return {
                    'skewness': None,
                    'kurtosis': None,
                    'normality_test': None,
                    'error': 'Insufficient data for analysis'
                }
                
            # Create visualization with error handling
            plt.figure(figsize=(10, 6))
            try:
                # Use kde=False if we have very few data points
                use_kde = kde and len(target_data) > 5
                sns.histplot(target_data, bins=min(bins, len(target_data) - 1), kde=use_kde)
                plt.title(f'Distribution of {target}')
                plt.savefig(os.path.join(self.output_dir, 'target_distribution.png'))
                plt.close()
                self.logger.info("Target distribution visualized and saved.")
            except Exception as e:
                plt.close()
                self.logger.warning(f"Error creating target distribution plot: {e}")
                
            # Calculate statistics with error handling
            try:
                skewness = target_data.skew()
                kurtosis = target_data.kurtosis()
                
                # Only perform Shapiro-Wilk test if we have enough data
                # (Shapiro-Wilk is reliable for sample sizes between 3 and 5000)
                if 3 <= len(target_data) <= 5000:
                    normality_test = stats.shapiro(target_data)
                    normality_result = {
                        'statistic': normality_test[0],
                        'pvalue': normality_test[1]
                    }
                else:
                    self.logger.warning("Sample size not suitable for Shapiro-Wilk test")
                    normality_result = {
                        'statistic': None,
                        'pvalue': None,
                        'note': 'Sample size not suitable for test'
                    }
                    
                distribution_stats = {
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'normality_test': normality_result
                }
                
                self.exploration_results['target_distribution'] = distribution_stats
                self.logger.info("Target distribution analysis completed.")
                return distribution_stats
                
            except Exception as e:
                self.logger.error(f"Error calculating target distribution statistics: {e}")
                return {
                    'skewness': None,
                    'kurtosis': None,
                    'normality_test': None,
                    'error': str(e)
                }
                
        except Exception as e:
            self.logger.error(f"Error in target distribution analysis: {e}")
            return {
                'error': str(e)
            }

    def identify_high_variance_features(self, threshold=0.1):
        """
        Detects features with low variance that may have limited predictive value.

        Args:
            threshold (float, optional): Variance threshold. Defaults to 0.1.

        Returns:
            list: Low-variance features.
        """
        low_variance_features = []
        for col in self.features_df.columns:
            if col in self.numeric_features:
                variance = self.features_df[col].var()
                if variance < threshold:
                    low_variance_features.append(col)
            else:
                # For categorical features, consider using a different metric like mode ratio
                mode_count = self.features_df[col].value_counts().iloc[0]
                mode_ratio = mode_count / len(self.features_df)
                if mode_ratio > (1 - threshold):  # High mode ratio implies low variance
                    low_variance_features.append(col)

        self.exploration_results['low_variance_features'] = low_variance_features
        self.logger.info("Low variance features identified.")
        return low_variance_features

    def generate_exploration_report(self, output_dir='reports', html_filename='data_exploration_report.html', md_filename='data_exploration_report.md', include_plots=True):
        """
        Compiles all exploration findings into comprehensive statistical reports.

        Args:
            output_dir (str, optional): Directory to save the report. Defaults to 'reports'.
            html_filename (str, optional): Filename for HTML report. Defaults to 'data_exploration_report.html'.
            md_filename (str, optional): Filename for Markdown report. Defaults to 'data_exploration_report.md'.
            include_plots (bool, optional): Whether to include plots in the report. Defaults to True.

        Returns:
            pd.DataFrame: Summary DataFrame containing key insights.
        """
        # Generate HTML report
        try:
            env = Environment(loader=FileSystemLoader('.'))
            template = env.get_template('report_template.html')  # Ensure you have a report_template.html

            report_data = {
                'basic_stats': self.exploration_results.get('basic_stats', None),
                'correlations': self.exploration_results.get('correlations', None),
                'outliers': self.exploration_results.get('outliers', None),
                'feature_importance': self.exploration_results.get('feature_importance', None),
                'missing_values': self.exploration_results.get('missing_values', None),
                'categorical_features': self.exploration_results.get('categorical_features', None),
                'target_distribution': self.exploration_results.get('target_distribution', None),
                'low_variance_features': self.exploration_results.get('low_variance_features', None),
                'include_plots': include_plots
            }

            html_output = template.render(report_data)

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            html_report_path = output_path / html_filename

            with open(html_report_path, 'w') as f:
                f.write(html_output)

            self.logger.info(f"HTML exploration report generated at {html_report_path}")
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
        
        # Generate Markdown report
        try:
            md_report_path = self.generate_markdown_report(output_dir, md_filename)
            self.logger.info(f"Markdown exploration report generated at {md_report_path}")
        except Exception as e:
            self.logger.error(f"Error generating Markdown report: {e}")

        # Create a summary DataFrame
        summary_data = {
            'Number of Features': [len(self.features_df.columns)],
            'Number of Numeric Features': [len(self.numeric_features)],
            'Number of Categorical Features': [len(self.categorical_features)],
            'Missing Values (%)': [self.features_df.isnull().sum().sum() / self.features_df.size * 100]
        }
        summary_df = pd.DataFrame(summary_data)

        return summary_df

    def generate_markdown_report(self, output_dir='reports', filename='data_exploration_report.md'):
        """
        Compiles all exploration findings into a comprehensive Markdown report.

        Args:
            output_dir (str, optional): Directory to save the report. Defaults to 'reports'.
            filename (str, optional): Filename for the report. Defaults to 'data_exploration_report.md'.

        Returns:
            str: Path to the generated report file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_path = output_path / filename
        
        with open(report_path, 'w') as f:
            # Title and introduction
            f.write("# Data Exploration Report\n\n")
            f.write("## Overview\n")
            f.write(f"* **Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"* **Number of Features:** {len(self.features_df.columns)}\n")
            f.write(f"* **Number of Numeric Features:** {len(self.numeric_features)}\n")
            f.write(f"* **Number of Categorical Features:** {len(self.categorical_features)}\n")
            f.write(f"* **Missing Values (%):** {self.features_df.isnull().sum().sum() / self.features_df.size * 100:.2f}%\n\n")
            
            # Basic statistics
            if 'basic_stats' in self.exploration_results and self.exploration_results['basic_stats'] is not None:
                f.write("## Basic Statistics\n\n")
                stats_md = self.exploration_results['basic_stats'].to_markdown()
                f.write(f"{stats_md}\n\n")
                
            # Feature importance
            if 'feature_importance' in self.exploration_results and self.exploration_results['feature_importance'] is not None:
                f.write("## Feature Importance\n\n")
                importance = self.exploration_results['feature_importance']
                f.write("### Top Important Features\n\n")
                importance_df = pd.DataFrame({'Importance': importance})
                importance_df = importance_df.sort_values('Importance', ascending=False)
                f.write(f"{importance_df.to_markdown()}\n\n")
                
                # Add a visualization reference
                f.write("![Feature Importance](feature_importance.png)\n\n")
                
            # Correlations
            if 'correlations' in self.exploration_results and self.exploration_results['correlations'] is not None:
                f.write("## Correlation Analysis\n\n")
                correlations = self.exploration_results['correlations']
                
                if 'high_corr_pairs' in correlations and correlations['high_corr_pairs']:
                    f.write("### Highly Correlated Feature Pairs\n\n")
                    corr_pairs = correlations['high_corr_pairs']
                    corr_df = pd.DataFrame(corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
                    f.write(f"{corr_df.to_markdown()}\n\n")
                    
                f.write("### Correlation Matrix\n\n")
                f.write("![Correlation Matrix](correlation_matrix_pearson.png)\n\n")
                
            # Missing values
            if 'missing_values' in self.exploration_results and self.exploration_results['missing_values'] is not None:
                f.write("## Missing Values Analysis\n\n")
                missing_df = self.exploration_results['missing_values']
                features_with_missing = missing_df[missing_df['Count'] > 0]
                
                if not features_with_missing.empty:
                    f.write(f"{features_with_missing.to_markdown()}\n\n")
                    f.write("### Missing Values Visualization\n\n")
                    f.write("![Missing Values Matrix](missing_values_matrix.png)\n\n")
                    f.write("![Missing Values Bar Chart](missing_values_bar.png)\n\n")
                else:
                    f.write("No missing values found in the dataset.\n\n")
                    
            # Outliers
            if 'outliers' in self.exploration_results and self.exploration_results['outliers'] is not None:
                f.write("## Outlier Analysis\n\n")
                outlier_stats = self.exploration_results['outliers']['stats']
                
                outlier_df = pd.DataFrame({
                    'Feature': list(outlier_stats.keys()),
                    'Count': [stats['count'] for stats in outlier_stats.values()],
                    'Percentage': [f"{stats['percentage']:.2f}%" for stats in outlier_stats.values()]
                })
                
                f.write(f"{outlier_df.to_markdown()}\n\n")
                
            # Target distribution
            if 'target_distribution' in self.exploration_results and self.exploration_results['target_distribution'] is not None:
                f.write("## Target Distribution Analysis\n\n")
                target_stats = self.exploration_results['target_distribution']
                
                if 'error' not in target_stats or target_stats['error'] is None:
                    f.write(f"* **Skewness:** {target_stats.get('skewness')}\n")
                    f.write(f"* **Kurtosis:** {target_stats.get('kurtosis')}\n")
                    
                    if 'normality_test' in target_stats and target_stats['normality_test'] is not None:
                        norm_test = target_stats['normality_test']
                        if 'pvalue' in norm_test and norm_test['pvalue'] is not None:
                            f.write(f"* **Shapiro-Wilk p-value:** {norm_test['pvalue']:.6f}\n")
                            f.write(f"* **Distribution is {'likely normal' if norm_test['pvalue'] > 0.05 else 'likely non-normal'}**\n\n")
                            
                    f.write("![Target Distribution](target_distribution.png)\n\n")
                else:
                    f.write(f"Error analyzing target distribution: {target_stats.get('error')}\n\n")
                    
            # Categorical features
            if 'categorical_features' in self.exploration_results and self.exploration_results['categorical_features'] is not None:
                f.write("## Categorical Features Analysis\n\n")
                cat_stats = self.exploration_results['categorical_features']
                
                cat_summary = []
                for col, stats in cat_stats.items():
                    cat_summary.append({
                        'Feature': col,
                        'Unique Values': stats['num_unique'],
                        'Entropy': f"{stats['entropy']:.4f}"
                    })
                    
                cat_df = pd.DataFrame(cat_summary)
                f.write(f"{cat_df.to_markdown()}\n\n")
                
                # Add visualizations for each categorical feature
                for col in cat_stats.keys():
                    f.write(f"### Distribution of {col}\n\n")
                    f.write(f"![{col} Distribution](categorical_{col}.png)\n\n")
                    
            # Low variance features
            if 'low_variance_features' in self.exploration_results and self.exploration_results['low_variance_features'] is not None:
                f.write("## Low Variance Features\n\n")
                low_var = self.exploration_results['low_variance_features']
                
                if low_var:
                    f.write("The following features have low variance and might have limited predictive value:\n\n")
                    for feature in low_var:
                        f.write(f"* {feature}\n")
                else:
                    f.write("No low variance features detected.\n\n")
                    
            # Conclusions
            f.write("## Conclusions and Recommendations\n\n")
            f.write("Based on the analysis, here are some key findings and recommendations:\n\n")
            
            # Add automated conclusions based on the data analysis results
            recommendations = []
            
            # Feature importance recommendations
            if 'feature_importance' in self.exploration_results and self.exploration_results['feature_importance'] is not None:
                importance = self.exploration_results['feature_importance']
                top_features = importance.nlargest(3).index.tolist()
                recommendations.append(f"* Consider focusing on the top important features: {', '.join(top_features)}")
            
            # Correlation recommendations
            if 'correlations' in self.exploration_results and self.exploration_results['correlations'] is not None:
                if 'high_corr_pairs' in self.exploration_results['correlations']:
                    high_corr = self.exploration_results['correlations']['high_corr_pairs']
                    if high_corr:
                        recommendations.append("* Consider removing some highly correlated features to reduce multicollinearity")
            
            # Missing values recommendations
            if 'missing_values' in self.exploration_results and self.exploration_results['missing_values'] is not None:
                missing_df = self.exploration_results['missing_values']
                high_missing = missing_df[missing_df['Percentage'] > 20].index.tolist()
                if high_missing:
                    recommendations.append(f"* Features with high missing values that might need special treatment: {', '.join(high_missing)}")
            
            # Target distribution recommendations
            if 'target_distribution' in self.exploration_results and self.exploration_results['target_distribution'] is not None:
                target_stats = self.exploration_results['target_distribution']
                if 'skewness' in target_stats and target_stats['skewness'] is not None:
                    skewness = target_stats['skewness']
                    if abs(skewness) > 1:
                        recommendations.append(f"* The target variable is {'positively' if skewness > 0 else 'negatively'} skewed (skewness = {skewness:.2f}). Consider applying a transformation.")
            
            # Write recommendations
            for rec in recommendations:
                f.write(f"{rec}\n")
                
            if not recommendations:
                f.write("* No specific recommendations based on the current analysis.\n")
                
        self.logger.info(f"Markdown report generated at {report_path}")
        return str(report_path)