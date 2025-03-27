import pandas as pd
import numpy as np
import os
from colorama import Fore, Style

class DataPreprocessor:
    """
    A class for preprocessing the district-wise key indicator dataset.
    Handles data loading, cleaning, and preparation for machine learning.
    """
    
    def __init__(self, file_path='Key_indicator_districtwise.csv'):
        """
        Initialize the DataPreprocessor with the CSV file path.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing the data
        """
        self.file_path = file_path
        self.df = None
        self.target_column = 'YY_Infant_Mortality_Rate_Imr_Total_Person'
        self.X = None  # Features
        self.y = None  # Target
        self.removed_columns = {
            'special_prefix': [],
            'high_missing': [],
            'demographic': []  # New category for demographic-filtered columns
        }
        self.original_shape = None
    
    def load_data(self):
        """
        Load data from the CSV file.
        
        Returns:
        --------
        pandas.DataFrame
            The loaded dataframe
        """
        print(f"Loading data from {self.file_path}...")
        self.df = pd.read_csv(self.file_path)
        self.original_shape = self.df.shape
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df
    
    def extract_target_variable(self):
        """
        Extract the target variable from the dataframe.
        
        Returns:
        --------
        pandas.Series
            The target variable
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print(f"Extracting target variable: {self.target_column}")
        
        # Check if target column exists
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in the dataframe.")
        
        self.y = self.df[self.target_column].copy()
        print(f"Target variable extracted. Shape: {self.y.shape}")
        
        return self.y
    
    def remove_special_columns(self):
        """
        Remove columns that start with 'ZZ' and 'YY' from the dataframe.
        
        Returns:
        --------
        pandas.DataFrame
            The cleaned dataframe with special columns removed
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        initial_column_count = len(self.df.columns)
        
        # Identify columns to remove
        columns_to_remove = [col for col in self.df.columns 
                            if col.startswith('ZZ') or col.startswith('YY')]
        
        print(f"Removing {len(columns_to_remove)} columns that start with 'ZZ' or 'YY'...")
        self.df = self.df.drop(columns=columns_to_remove)
        
        # Store removed columns
        self.removed_columns['special_prefix'] = columns_to_remove
        
        print(f"Columns removed. Original count: {initial_column_count}, New count: {len(self.df.columns)}")
        return self.df
    
    def remove_high_missing_columns(self, threshold=75):
        """
        Remove columns with a high percentage of missing values.
        
        Parameters:
        -----------
        threshold : float
            The percentage threshold above which a column will be removed (default: 75%)
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with high-missing columns removed
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        initial_column_count = len(self.df.columns)
        
        # Calculate percentage of missing values for each column
        missing_percentages = 100 * self.df.isnull().sum() / len(self.df)
        
        # Identify columns to remove
        high_missing_cols = missing_percentages[missing_percentages > threshold].index.tolist()
        
        if high_missing_cols:
            print(f"{Fore.YELLOW}⚠ Removing {len(high_missing_cols)} columns with >{threshold}% missing values:{Style.RESET_ALL}")
            
            # Print list of columns being removed (with truncation if too many)
            if len(high_missing_cols) <= 10:
                for col in high_missing_cols:
                    print(f"  - {col} ({missing_percentages[col]:.1f}% missing)")
            else:
                for col in high_missing_cols[:5]:
                    print(f"  - {col} ({missing_percentages[col]:.1f}% missing)")
                print(f"  - ... and {len(high_missing_cols) - 5} more columns")
            
            # Remove the columns
            self.df = self.df.drop(columns=high_missing_cols)
            
            # Store removed columns
            self.removed_columns['high_missing'] = high_missing_cols
            
            print(f"{Fore.GREEN}✓ Columns removed. Original count: {initial_column_count}, New count: {len(self.df.columns)}{Style.RESET_ALL}")
        else:
            print(f"{Fore.BLUE}ℹ No columns have >{threshold}% missing values.{Style.RESET_ALL}")
            
        return self.df
    
    def remove_demographic_columns(self, demographics=None):
        """
        Remove columns that contain specific demographic indicators.
        
        Parameters:
        -----------
        demographics : list
            List of demographic indicators to filter out (e.g., ['Rural', 'Urban'])
            If None, user will be asked to choose which demographics to filter
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with demographic columns removed
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        initial_column_count = len(self.df.columns)
        
        # If demographics not provided, ask user
        if demographics is None:
            demographics = self.ask_demographic_filters()
        
        if not demographics:
            print(f"{Fore.BLUE}ℹ No demographic filters selected. Skipping demographic filtering.{Style.RESET_ALL}")
            return self.df
        
        # Define case-insensitive patterns for each demographic
        patterns = [f"(?i){demo}" for demo in demographics]
        
        # Identify columns that match any of the patterns
        columns_to_remove = []
        for pattern in patterns:
            matched_cols = [col for col in self.df.columns if pattern.lower().replace("(?i)", "") in col.lower()]
            columns_to_remove.extend(matched_cols)
        
        # Remove duplicates
        columns_to_remove = list(set(columns_to_remove))
        
        if columns_to_remove:
            print(f"{Fore.YELLOW}⚠ Removing {len(columns_to_remove)} columns containing demographic indicators: {', '.join(demographics)}{Style.RESET_ALL}")
            
            # Print list of columns being removed (with truncation if too many)
            if len(columns_to_remove) <= 10:
                for col in columns_to_remove:
                    print(f"  - {col}")
            else:
                for col in columns_to_remove[:5]:
                    print(f"  - {col}")
                print(f"  - ... and {len(columns_to_remove) - 5} more columns")
            
            # Remove the columns
            self.df = self.df.drop(columns=columns_to_remove)
            
            # Store removed columns
            self.removed_columns['demographic'] = columns_to_remove
            
            print(f"{Fore.GREEN}✓ Columns removed. Original count: {initial_column_count}, New count: {len(self.df.columns)}{Style.RESET_ALL}")
        else:
            print(f"{Fore.BLUE}ℹ No columns match the selected demographic indicators.{Style.RESET_ALL}")
            
        return self.df
    
    def ask_demographic_filters(self):
        """
        Ask user which demographic indicators to filter out.
        
        Returns:
        --------
        list
            List of selected demographic indicators
        """
        demographics = ['Rural', 'Urban', 'Male', 'Female']
        selected = []
        
        print(f"\n{Fore.CYAN}Select demographic indicators to filter out (columns containing these terms will be removed):{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[1]{Style.RESET_ALL} Rural")
        print(f"{Fore.YELLOW}[2]{Style.RESET_ALL} Urban")
        print(f"{Fore.YELLOW}[3]{Style.RESET_ALL} Male")
        print(f"{Fore.YELLOW}[4]{Style.RESET_ALL} Female")
        print(f"{Fore.YELLOW}[5]{Style.RESET_ALL} All of the above")
        print(f"{Fore.YELLOW}[6]{Style.RESET_ALL} None (keep all demographic indicators)")
        
        while True:
            try:
                choice = input(f"\n{Fore.CYAN}Enter your choice (1-6 or comma-separated list e.g. 1,3,4): {Style.RESET_ALL}")
                
                if choice == '5':
                    return demographics
                elif choice == '6':
                    return []
                elif ',' in choice:
                    # Handle comma-separated list
                    indices = [int(idx.strip()) for idx in choice.split(',')]
                    for idx in indices:
                        if 1 <= idx <= 4:
                            selected.append(demographics[idx-1])
                        else:
                            print(f"{Fore.RED}✗ Invalid choice: {idx}. Please enter numbers between 1 and 6.{Style.RESET_ALL}")
                            continue
                    return selected
                elif choice in ['1', '2', '3', '4']:
                    selected.append(demographics[int(choice)-1])
                    return selected
                else:
                    print(f"{Fore.RED}✗ Invalid choice. Please enter a number between 1 and 6 or a comma-separated list.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}✗ Please enter valid number(s).{Style.RESET_ALL}")
    
    def prepare_features(self):
        """
        Prepare the feature set for machine learning by dropping the target column.
        
        Returns:
        --------
        pandas.DataFrame
            The feature set (X)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if self.y is None:
            # If target hasn't been extracted yet, extract it first
            self.extract_target_variable()
            
        # Create feature set (X) by excluding the target column
        self.X = self.df.copy()
        
        if self.target_column in self.X.columns:
            self.X = self.X.drop(columns=[self.target_column])
        
        print(f"Features prepared. Feature set shape: {self.X.shape}")
        return self.X
    
    def handle_missing_values(self, strategy='median'):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        strategy : str
            Strategy to handle missing values ('median', 'mean', 'mode', 'drop')
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with handled missing values
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        print(f"Handling missing values using strategy: {strategy}")
        
        # Count missing values before handling
        missing_before = self.df.isnull().sum().sum()
        
        if strategy == 'drop':
            # Drop rows with any missing values
            self.df = self.df.dropna()
            
        elif strategy in ['median', 'mean', 'mode']:
            # Handle numeric columns
            numeric_cols = self.df.select_dtypes(include=np.number).columns
            
            for col in numeric_cols:
                if strategy == 'median':
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif strategy == 'mean':
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif strategy == 'mode':
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            
            # Handle categorical columns with mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        else:
            raise ValueError("Invalid strategy. Use 'median', 'mean', 'mode', or 'drop'.")
        
        # Count missing values after handling
        missing_after = self.df.isnull().sum().sum()
        
        print(f"Missing values before: {missing_before}, after: {missing_after}")
        return self.df
    
    def ask_missing_values_strategy(self):
        """
        Ask user for missing values handling strategy.
        
        Returns:
        --------
        str
            The selected strategy ('median', 'mean', 'mode', or 'drop')
        """
        print(f"\n{Fore.CYAN}Choose a strategy for handling missing values:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[1]{Style.RESET_ALL} Median (default, recommended for skewed numeric data)")
        print(f"{Fore.YELLOW}[2]{Style.RESET_ALL} Mean (good for normally distributed numeric data)")
        print(f"{Fore.YELLOW}[3]{Style.RESET_ALL} Mode (most frequent value, used for categorical data)")
        print(f"{Fore.YELLOW}[4]{Style.RESET_ALL} Drop (remove rows with missing values - not recommended if many missing)")
        
        while True:
            try:
                choice = input(f"\n{Fore.CYAN}Enter your choice (1-4): {Style.RESET_ALL}")
                
                if choice == '1':
                    return 'median'
                elif choice == '2':
                    return 'mean'
                elif choice == '3':
                    return 'mode'
                elif choice == '4':
                    return 'drop'
                else:
                    print(f"{Fore.RED}✗ Invalid choice. Please enter a number between 1 and 4.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}✗ Please enter a valid number.{Style.RESET_ALL}")
    
    def print_summary(self):
        """
        Print summary of preprocessing steps and removed columns.
        """
        print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{Style.BRIGHT}PREPROCESSING SUMMARY{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        
        # Original dataset info
        print(f"\n{Fore.WHITE}{Style.BRIGHT}Original Dataset:{Style.RESET_ALL}")
        print(f"Rows: {self.original_shape[0]}")
        print(f"Columns: {self.original_shape[1]}")
        
        # Removed columns summary
        print(f"\n{Fore.WHITE}{Style.BRIGHT}Removed Columns:{Style.RESET_ALL}")
        
        # Special prefix columns
        special_count = len(self.removed_columns['special_prefix'])
        print(f"\n1. Special prefix columns (ZZ*, YY*): {special_count}")
        if special_count > 0:
            if special_count <= 10:
                for col in self.removed_columns['special_prefix']:
                    print(f"  - {col}")
            else:
                for col in self.removed_columns['special_prefix'][:5]:
                    print(f"  - {col}")
                print(f"  - ... and {special_count - 5} more columns")
                
        # High missing columns
        high_missing_count = len(self.removed_columns['high_missing'])
        print(f"\n2. High missing value columns: {high_missing_count}")
        if high_missing_count > 0:
            if high_missing_count <= 10:
                for col in self.removed_columns['high_missing']:
                    print(f"  - {col}")
            else:
                for col in self.removed_columns['high_missing'][:5]:
                    print(f"  - {col}")
                print(f"  - ... and {high_missing_count - 5} more columns")
        
        # Demographic columns
        demographic_count = len(self.removed_columns['demographic'])
        print(f"\n3. Demographic indicator columns: {demographic_count}")
        if demographic_count > 0:
            if demographic_count <= 10:
                for col in self.removed_columns['demographic']:
                    print(f"  - {col}")
            else:
                for col in self.removed_columns['demographic'][:5]:
                    print(f"  - {col}")
                print(f"  - ... and {demographic_count - 5} more columns")
        
        # Final datasets
        print(f"\n{Fore.WHITE}{Style.BRIGHT}Final Datasets:{Style.RESET_ALL}")
        print(f"Features (X): {self.X.shape[0]} rows, {self.X.shape[1]} columns")
        print(f"Target (y): {self.y.shape[0]} elements")
        
        # Total columns removed
        total_removed = special_count + high_missing_count + demographic_count
        print(f"\n{Fore.WHITE}{Style.BRIGHT}Total columns removed:{Style.RESET_ALL} {total_removed}")
        print(f"Original columns: {self.original_shape[1]}")
        print(f"Final columns: {self.X.shape[1] + 1}") # +1 for target column
        
        print(f"\n{Fore.GREEN}✓ Preprocessing completed successfully!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

    def preprocess(self, remove_high_missing=True, high_missing_threshold=75, filter_demographics=True):
        """
        Execute the full preprocessing pipeline.
        
        Parameters:
        -----------
        remove_high_missing : bool
            Whether to remove columns with high missing values (default: True)
        high_missing_threshold : float
            The threshold percentage for removing high missing columns (default: 75%)
        filter_demographics : bool
            Whether to filter columns based on demographic indicators (default: True)
            
        Returns:
        --------
        tuple
            A tuple containing (X, y) - the features and target
        """
        print(f"{Fore.CYAN}Starting preprocessing pipeline...{Style.RESET_ALL}")
        
        # Execute preprocessing steps
        self.load_data()
        self.extract_target_variable()
        self.remove_special_columns()
        
        if remove_high_missing:
            self.remove_high_missing_columns(threshold=high_missing_threshold)
        
        if filter_demographics:
            self.remove_demographic_columns()
        
        # Ask for missing values strategy
        strategy = self.ask_missing_values_strategy()    
        self.handle_missing_values(strategy=strategy)
        self.prepare_features()
        
        # Print summary of preprocessing steps and removed columns
        self.print_summary()
        
        return self.X, self.y


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor('Key_indicator_districtwise.csv')
    X, y = preprocessor.preprocess()
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save preprocessed data to data folder
    X_path = os.path.join(data_dir, 'preprocessed_features.csv')
    y_path = os.path.join(data_dir, 'preprocessed_target.csv')
    
    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False)
    print(f"\nPreprocessed data saved to data folder.")