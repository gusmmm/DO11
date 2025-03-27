"""
Module: health_data_analyzer.py
Purpose: Provides a streamlined analysis tool to explore health data, identify meaningful features, and provide recommendations for machine learning model selection. Designed to handle large datasets with mixed data types efficiently, without complex computations.
Dependencies:
    pandas: For efficient data manipulation and analysis.
    numpy: For numerical operations.
    matplotlib: For basic plotting.
    seaborn: For enhanced visualizations.
    sklearn: For basic feature selection tools.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import logging
from pathlib import Path
from datetime import datetime
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)

# Constants
DEFAULT_NUMERIC_FEATURES = ['population', 'rate', 'ratio', 'percentage', 'number', 'mean', 'median', 'std', 'score']
DEFAULT_CATEGORICAL_FEATURES = ['state', 'region', 'district']
COLOR_SCHEMES = {
    'header': {'fore': Fore.CYAN, 'style': Style.BRIGHT},
    'section': {'fore': Fore.GREEN, 'style': Style.BRIGHT},
    'subsection': {'fore': Fore.YELLOW, 'style': Style.BRIGHT},
    'recommendation': {'fore': Fore.MAGENTA, 'style': Style.BRIGHT},
    'info': {'fore': Fore.WHITE, 'style': Style.NORMAL},
    'error': {'fore': Fore.RED, 'style': Style.BRIGHT}
}

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Utility functions
def safe_convert_numeric(series):
    """Safely converts a pandas Series to numeric, handling errors."""
    try:
        return pd.to_numeric(series, errors='coerce')
    except Exception as e:
        logger.warning(f"Could not convert series to numeric: {e}")
        return series

def print_formatted_output(text, style='info'):
    """Prints formatted text to the terminal with color coding."""
    color_settings = COLOR_SCHEMES.get(style, COLOR_SCHEMES['info'])
    print(f"{color_settings['fore']}{color_settings['style']}{text}{Style.RESET_ALL}")

class HealthDataAnalyzer:
    """
    Analyzes health data for feature selection and provides ML recommendations.
    """
    def __init__(self, features_df, target_df, target_column=None, output_dir='reports'):
        """
        Initializes the analyzer with data and configuration options.

        Args:
            features_df (pd.DataFrame): DataFrame containing feature data.
            target_df (pd.DataFrame): DataFrame containing target variable data.
            target_column (str, optional): Name of the target column. Defaults to None.
            output_dir (str, optional): Directory for saving reports. Defaults to 'reports'.
        """
        self.features_df = features_df.copy()
        self.target_df = target_df.copy()
        self.target_column = target_column
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.feature_types = {}
        self.analysis_results = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.preprocess_data()
        self.detect_feature_types()
        self.logger.info("HealthDataAnalyzer initialized.")

    def preprocess_data(self):
        """Performs basic preprocessing to ensure data quality."""
        print_formatted_output("Preprocessing data...", style='section')
        # Convert column names to lowercase and replace spaces with underscores
        self.features_df.columns = self.features_df.columns.str.lower().str.replace(' ', '_')
        self.target_df.columns = self.target_df.columns.str.lower().str.replace(' ', '_')

        # Attempt to convert all columns to numeric where possible
        for col in self.features_df.columns:
            self.features_df[col] = safe_convert_numeric(self.features_df[col])
        for col in self.target_df.columns:
            self.target_df[col] = safe_convert_numeric(self.target_df[col])

        # Remove any columns with object dtype that couldn't be converted
        object_cols = self.features_df.select_dtypes(include=['object']).columns
        if not object_cols.empty:
            print_formatted_output(f"Dropping non-numeric columns: {', '.join(object_cols)}", style='info')
            self.features_df.drop(columns=object_cols, inplace=True, errors='ignore')

        # Drop any rows with missing values in the target variable
        if self.target_column in self.target_df.columns:
            self.target_df.dropna(subset=[self.target_column], inplace=True)
            self.logger.info(f"Dropped rows with missing values in target column: {self.target_column}")
        else:
            self.logger.warning("Target column not found in target_df, skipping NaN removal.")

        self.logger.info("Data preprocessing complete.")

    def detect_feature_types(self):
        """Detects feature types based on column names and data content."""
        print_formatted_output("Detecting feature types...", style='section')
        for col in self.features_df.columns:
            if any(keyword in col.lower() for keyword in DEFAULT_CATEGORICAL_FEATURES):
                self.feature_types[col] = 'categorical'
            elif any(keyword in col.lower() for keyword in DEFAULT_NUMERIC_FEATURES):
                self.feature_types[col] = 'numeric'
            elif pd.api.types.is_numeric_dtype(self.features_df[col]):
                self.feature_types[col] = 'numeric'
            else:
                self.feature_types[col] = 'categorical'
        self.logger.info(f"Detected feature types: {self.feature_types}")

    def get_basic_feature_stats(self, n_features=20):
        """Calculates basic statistics for top features."""
        print_formatted_output("Calculating basic feature statistics...", style='section')
        numeric_cols = [col for col, type in self.feature_types.items() if type == 'numeric']
        if not numeric_cols:
            self.logger.warning("No numeric columns found.")
            return pd.DataFrame()

        stats = pd.DataFrame()
        for col in numeric_cols[:n_features]:
            try:
                stats[col + '_mean'] = [self.features_df[col].mean()]
                stats[col + '_median'] = [self.features_df[col].median()]
                stats[col + '_std'] = [self.features_df[col].std()]
            except Exception as e:
                self.logger.error(f"Error calculating statistics for column {col}: {e}")
        self.analysis_results['basic_stats'] = stats
        self.logger.info("Basic feature statistics calculated.")
        return stats

    def analyze_correlations(self, threshold=0.7, method='spearman', max_features=30):
        """Analyzes feature correlations using Spearman."""
        print_formatted_output("Analyzing feature correlations...", style='section')
        numeric_cols = [col for col, type in self.feature_types.items() if type == 'numeric']
        if len(numeric_cols) > max_features:
            numeric_cols = numeric_cols[:max_features]
            self.logger.info(f"Limiting correlation analysis to top {max_features} features.")

        corr_matrix = self.features_df[numeric_cols].corr(method=method)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

        self.analysis_results['correlations'] = {'corr_matrix': corr_matrix, 'high_corr_pairs': high_corr_pairs}
        self.logger.info("Feature correlations analyzed.")
        return corr_matrix, high_corr_pairs

    def identify_meaningful_features(self, n_features=20):
        """Identifies potentially meaningful features based on variance and correlation."""
        print_formatted_output("Identifying meaningful features...", style='section')
        numeric_cols = [col for col, type in self.feature_types.items() if type == 'numeric']
        if not numeric_cols:
            self.logger.warning("No numeric columns found.")
            return []

        # Calculate variance
        variances = self.features_df[numeric_cols].var().sort_values(ascending=False)
        meaningful_features = variances.head(n_features).index.tolist()
        self.analysis_results['meaningful_features'] = meaningful_features
        self.logger.info("Meaningful features identified.")
        return meaningful_features

    def analyze_target_distribution(self):
        """Analyzes the distribution of the target variable."""
        print_formatted_output("Analyzing target distribution...", style='section')
        if self.target_column not in self.target_df.columns:
            self.logger.error(f"Target column '{self.target_column}' not found.")
            return {}

        target_data = self.target_df[self.target_column].dropna()
        if target_data.empty:
            self.logger.warning("Target data is empty after dropping NaNs.")
            return {}

        skewness = target_data.skew()
        kurtosis = target_data.kurtosis()
        self.analysis_results['target_distribution'] = {'skewness': skewness, 'kurtosis': kurtosis}
        self.logger.info("Target distribution analyzed.")
        return {'skewness': skewness, 'kurtosis': kurtosis}

    def analyze_missing_values(self, threshold=0.3):
        """Identifies features with significant missing values."""
        print_formatted_output("Analyzing missing values...", style='section')
        missing_values = self.features_df.isnull().sum() / len(self.features_df)
        missing_values = missing_values[missing_values > threshold].sort_values(ascending=False)
        self.analysis_results['missing_values'] = missing_values
        self.logger.info("Missing values analyzed.")
        return missing_values

    def generate_feature_selection_report(self, output_format='markdown'):
        """Generates a concise report focused on helping feature selection."""
        print_formatted_output("Generating feature selection report...", style='section')
        report = f"# Feature Selection Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "## Basic Stats\n"
        if 'basic_stats' in self.analysis_results and not self.analysis_results['basic_stats'].empty:
            report += self.analysis_results['basic_stats'].to_string() + "\n\n"
        else:
            report += "No basic statistics calculated.\n\n"

        report += "## Meaningful Features\n"
        if 'meaningful_features' in self.analysis_results:
            report += ", ".join(self.analysis_results['meaningful_features']) + "\n\n"
        else:
            report += "No meaningful features identified.\n\n"

        report += "## Correlations\n"
        if 'correlations' in self.analysis_results and self.analysis_results['correlations']['high_corr_pairs']:
            report += "High Correlation Pairs:\n"
            for pair in self.analysis_results['correlations']['high_corr_pairs']:
                report += f"- {pair[0]} and {pair[1]}: {pair[2]:.2f}\n"
        else:
            report += "No high correlation pairs found.\n\n"

        report += "## Missing Values\n"
        if 'missing_values' in self.analysis_results and not self.analysis_results['missing_values'].empty:
            report += self.analysis_results['missing_values'].to_string() + "\n\n"
        else:
            report += "No significant missing values found.\n\n"

        report_path = Path(self.output_dir) / 'feature_selection_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        self.logger.info(f"Feature selection report generated at {report_path}")
        return report

    def plot_top_features(self, n_features=10):
        """Creates simple visualizations of the most important features."""
        print_formatted_output("Plotting top features...", style='section')
        if 'meaningful_features' not in self.analysis_results:
            self.logger.warning("No meaningful features found to plot.")
            return

        top_features = self.analysis_results['meaningful_features'][:n_features]
        if not top_features:
            self.logger.warning("No top features to plot.")
            return

        plt.figure(figsize=(10, 6))
        self.features_df[top_features].boxplot(vert=False)
        plt.title('Top Features Distribution')
        plt.tight_layout()
        plot_path = Path(self.output_dir) / 'top_features_boxplot.png'
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Top features plot saved to {plot_path}")

    def recommend_models(self):
        """Provides basic recommendations for machine learning models."""
        print_formatted_output("Recommending machine learning models...", style='section')
        recommendations = []
        if 'target_distribution' in self.analysis_results:
            skewness = self.analysis_results['target_distribution'].get('skewness', 0)
            if abs(skewness) > 0.5:
                recommendations.append("Consider non-linear models or target transformation due to skewness.")
            else:
                recommendations.append("Linear models may be suitable for the target variable.")
        if 'correlations' in self.analysis_results and self.analysis_results['correlations']['high_corr_pairs']:
            recommendations.append("Consider regularization techniques to handle multicollinearity.")

        print_formatted_output("Recommended Models:\n" + "\n".join(recommendations), style='recommendation')

def run_analysis(features_path, target_path, target_column, output_dir='reports', feature_limit=50):
    """
    Top-level function that orchestrates the analysis process.

    Args:
        features_path (str): Path to the features CSV file.
        target_path (str): Path to the target CSV file.
        output_dir (str): Directory to save analysis reports.
        feature_limit (int): Limit the number of features to analyze.
    """
    print_formatted_output("Starting health data analysis...", style='header')
    try:
        features_df = pd.read_csv(features_path)
        target_df = pd.read_csv(target_path)
        analyzer = HealthDataAnalyzer(features_df, target_df, target_column, output_dir)

        analyzer.get_basic_feature_stats(n_features=feature_limit)
        analyzer.analyze_correlations()
        analyzer.identify_meaningful_features()
        analyzer.analyze_target_distribution()
        analyzer.analyze_missing_values()
        analyzer.plot_top_features()
        analyzer.recommend_models()
        analyzer.generate_feature_selection_report()

        print_formatted_output(f"Analysis complete. Reports saved to {output_dir}", style='info')

    except Exception as e:
        print_formatted_output(f"An error occurred: {e}", style='error')

if __name__ == '__main__':
    # Example usage
    features_path = 'data/preprocessed_features_20250327_135655.csv'
    target_path = 'data/preprocessed_target_20250327_135655.csv'
    target_column = 'YY_Infant_Mortality_Rate_Imr_Total_Person'
    output_dir = 'analysis_output'
    run_analysis(features_path, target_path, target_column, output_dir)