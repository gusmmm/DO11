import os
import sys
import glob
from tabulate import tabulate
import pandas as pd
import numpy as np
import time
from colorama import init, Fore, Style

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessor import DataPreprocessor

class MenuController:
    """
    Controls the interactive menu for the data science workflow.
    Provides options for data loading, preprocessing, analysis, and modeling.
    """
    
    def __init__(self):
        """Initialize the menu controller with necessary attributes."""
        init()  # Initialize colorama for colored terminal output
        self.preprocessor = None
        self.df = None
        self.X = None
        self.y = None
        self.data_loaded = False
        self.data_preprocessed = False
        self.models = {}
        self.csv_path = None
        
    def clear_screen(self):
        """Clear the terminal screen based on the operating system."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_header(self, title):
        """Print a formatted header for each menu section."""
        self.clear_screen()
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{Style.BRIGHT}{title.center(80)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        print()
        
    def print_success(self, message):
        """Print a success message."""
        print(f"\n{Fore.GREEN}✓ {message}{Style.RESET_ALL}\n")
        
    def print_error(self, message):
        """Print an error message."""
        print(f"\n{Fore.RED}✗ {message}{Style.RESET_ALL}\n")
        
    def print_warning(self, message):
        """Print a warning message."""
        print(f"\n{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}\n")
        
    def print_info(self, message):
        """Print an informational message."""
        print(f"\n{Fore.BLUE}ℹ {message}{Style.RESET_ALL}\n")
        
    def pause(self):
        """Pause execution until user presses Enter."""
        input(f"\n{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")
    
    def select_data_file(self):
        """Allow user to select a CSV file from the data directory."""
        self.print_header("SELECT DATA FILE")
        
        try:
            # Get all CSV files in the data directory
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
            csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
            
            if not csv_files:
                self.print_error("No CSV files found in the data directory!")
                return None
            
            print(f"{Fore.WHITE}{Style.BRIGHT}Available CSV files:{Style.RESET_ALL}\n")
            
            # Display available files with indices
            for i, file_path in enumerate(csv_files, 1):
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) / 1024  # Size in KB
                print(f"{Fore.YELLOW}[{i}]{Style.RESET_ALL} {file_name} ({file_size:.1f} KB)")
            
            # Get user choice
            while True:
                try:
                    choice = int(input(f"\n{Fore.CYAN}Enter file number to load (1-{len(csv_files)}): {Style.RESET_ALL}"))
                    if 1 <= choice <= len(csv_files):
                        selected_file = csv_files[choice - 1]
                        self.print_success(f"Selected: {os.path.basename(selected_file)}")
                        return selected_file
                    else:
                        self.print_error(f"Please enter a number between 1 and {len(csv_files)}")
                except ValueError:
                    self.print_error("Please enter a valid number")
                    
        except Exception as e:
            self.print_error(f"Error selecting file: {str(e)}")
            return None
        
    def load_data(self):
        """Load the CSV data and display basic information."""
        self.print_header("DATA LOADING")
        
        try:
            # Let user select a file
            selected_file = self.select_data_file()
            if not selected_file:
                self.print_error("File selection cancelled or failed.")
                self.pause()
                return
            
            self.csv_path = selected_file
            print(f"{Fore.BLUE}Loading data from: {os.path.basename(self.csv_path)}{Style.RESET_ALL}")
            
            # Create an instance of DataPreprocessor
            self.preprocessor = DataPreprocessor(self.csv_path)
            self.df = self.preprocessor.load_data()
            
            # Display dataset shape
            print(f"\n{Fore.GREEN}Dataset successfully loaded!{Style.RESET_ALL}")
            print(f"\n{Fore.WHITE}{Style.BRIGHT}Dataset Shape:{Style.RESET_ALL}")
            print(f"Rows: {self.df.shape[0]}")
            print(f"Columns: {self.df.shape[1]}")
            
            # Create concise variable summary
            print(f"\n{Fore.WHITE}{Style.BRIGHT}Variable Summary:{Style.RESET_ALL}")
            
            # Create a summary table
            summary_data = []
            for col in self.df.columns:
                col_type = self.df[col].dtype
                missing = self.df[col].isna().sum()
                missing_pct = 100 * missing / len(self.df)
                
                # Column info for all variables
                col_info = {
                    'Variable': col,
                    'Type': str(col_type),
                    'Missing': f"{missing} ({missing_pct:.1f}%)"
                }
                
                # Add statistics based on type
                if np.issubdtype(col_type, np.number):
                    col_info['Categories'] = 'N/A'
                    if missing < len(self.df):  # Only calculate stats if not all values are missing
                        col_info['Mean'] = f"{self.df[col].mean():.2f}"
                        col_info['Median'] = f"{self.df[col].median():.2f}"
                        col_info['Min'] = f"{self.df[col].min():.2f}"
                        col_info['Max'] = f"{self.df[col].max():.2f}"
                        col_info['Std'] = f"{self.df[col].std():.2f}"
                    else:
                        col_info['Mean'] = 'N/A'
                        col_info['Median'] = 'N/A'
                        col_info['Min'] = 'N/A'
                        col_info['Max'] = 'N/A'
                        col_info['Std'] = 'N/A'
                else:
                    # Categorical variable
                    n_categories = self.df[col].nunique()
                    col_info['Categories'] = n_categories
                    col_info['Mean'] = 'N/A'
                    col_info['Median'] = 'N/A'
                    col_info['Min'] = 'N/A'
                    col_info['Max'] = 'N/A'
                    col_info['Std'] = 'N/A'
                
                summary_data.append(col_info)
            
            # Display summary table (first 10 columns with option to see more)
            summary_df = pd.DataFrame(summary_data)
            print(tabulate(summary_df.head(10), headers='keys', tablefmt='pretty', showindex=False))
            if len(summary_df) > 10:
                print(f"...showing 10 of {len(summary_df)} variables")
                
                # Ask if user wants to see all variables
                see_all = input(f"\n{Fore.CYAN}Do you want to see all variables? (y/n): {Style.RESET_ALL}").lower()
                if see_all == 'y':
                    print("\n" + tabulate(summary_df, headers='keys', tablefmt='pretty', showindex=False))
                
            # Separate table for variables with missing values
            missing_cols = summary_df[summary_df['Missing'].str.contains('0 \(0.0%\)') == False].copy()
            
            if not missing_cols.empty:
                print(f"\n{Fore.WHITE}{Style.BRIGHT}Variables with Missing Values:{Style.RESET_ALL}")
                print(tabulate(missing_cols, headers='keys', tablefmt='pretty', showindex=False))
                print(f"\nTotal columns with missing values: {len(missing_cols)}")
                
                # Summary statistics for missing values
                missing_counts = self.df.isna().sum()
                missing_counts = missing_counts[missing_counts > 0]
                
                print(f"\n{Fore.WHITE}{Style.BRIGHT}Missing Values Statistics:{Style.RESET_ALL}")
                print(f"Min missing values in a column: {missing_counts.min()}")
                print(f"Max missing values in a column: {missing_counts.max()}")
                print(f"Mean missing values per column: {missing_counts.mean():.1f}")
                
            # Create a dedicated missing values analysis - simplified and ordered
            missing_counts = self.df.isnull().sum()
            missing_pcts = 100 * missing_counts / len(self.df)
            
            # Create a dataframe with only columns that have missing values
            missing_data = pd.DataFrame({
                'Missing Count': missing_counts,
                'Missing %': missing_pcts.round(1)
            })
            missing_data = missing_data[missing_data['Missing Count'] > 0]
            
            # Sort by missing count in descending order
            missing_data = missing_data.sort_values('Missing Count', ascending=False)
            
            if not missing_data.empty:
                print(f"\n{Fore.WHITE}{Style.BRIGHT}Variables with Missing Values (Top 10):{Style.RESET_ALL}")
                
                # Display only the top 10 columns with most missing values
                top_missing = missing_data.head(10).copy()
                
                # Add variable name as the first column instead of index
                display_df = pd.DataFrame({
                    'Variable': top_missing.index,
                    'Missing Count': top_missing['Missing Count'],
                    'Missing %': top_missing['Missing %'].apply(lambda x: f"{x}%")
                })
                
                # Print the table with clear formatting
                print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
                
                print(f"\nShowing top 10 of {len(missing_data)} columns with missing values")
                
                # Display missing values summary statistics
                print(f"\n{Fore.WHITE}{Style.BRIGHT}Missing Values Summary:{Style.RESET_ALL}")
                print(f"Total columns with missing values: {len(missing_data)} out of {len(self.df.columns)}")
                print(f"Min missing values in a column: {missing_data['Missing Count'].min()}")
                print(f"Max missing values in a column: {missing_data['Missing Count'].max()}")
                print(f"Mean missing values per column: {missing_data['Missing Count'].mean():.1f}")
                
                # Ask if user wants to see all missing columns
                if len(missing_data) > 10:
                    see_all = input(f"\n{Fore.CYAN}Do you want to see all columns with missing values? (y/n): {Style.RESET_ALL}").lower()
                    if see_all == 'y':
                        # Create full missing values table, still with simpler format
                        full_display_df = pd.DataFrame({
                            'Variable': missing_data.index,
                            'Missing Count': missing_data['Missing Count'],
                            'Missing %': missing_data['Missing %'].apply(lambda x: f"{x}%")
                        })
                        print("\n" + tabulate(full_display_df, headers='keys', tablefmt='grid', showindex=False))
            else:
                print(f"\n{Fore.GREEN}No missing values found in the dataset.{Style.RESET_ALL}")
                
            # Display data types distribution
            print(f"\n{Fore.WHITE}{Style.BRIGHT}Data Types Distribution:{Style.RESET_ALL}")
            dtypes_count = self.df.dtypes.value_counts().reset_index()
            dtypes_count.columns = ['Data Type', 'Count']
            print(tabulate(dtypes_count, headers='keys', tablefmt='pretty', showindex=False))
            
            self.data_loaded = True
            self.print_success("Data loaded and analyzed successfully!")
            
        except Exception as e:
            self.print_error(f"Error loading data: {str(e)}")
            
        self.pause()
        
    def preprocess_data(self):
        """Preprocess the data using the DataPreprocessor class."""
        if not self.data_loaded:
            self.print_error("Please load the data first!")
            self.pause()
            return
            
        self.print_header("DATA PREPROCESSING")
        
        try:
            print("Select preprocessing options:")
            print(f"\n{Fore.YELLOW}[1]{Style.RESET_ALL} Standard preprocessing")
            print(f"{Fore.YELLOW}[2]{Style.RESET_ALL} Standard preprocessing + Remove high-missing columns (>75%)")
            print(f"{Fore.YELLOW}[0]{Style.RESET_ALL} Cancel")
            
            while True:
                try:
                    choice = input(f"\n{Fore.CYAN}Enter your choice (0-2): {Style.RESET_ALL}")
                    
                    if choice == '0':
                        self.print_info("Preprocessing cancelled.")
                        return
                    elif choice in ['1', '2']:
                        break
                    else:
                        self.print_error("Invalid choice. Please enter a number between 0 and 2.")
                except ValueError:
                    self.print_error("Please enter a valid number")
            
            # Define preprocessing parameters based on user choice
            remove_high_missing = (choice == '2')
            
            # Print selected preprocessing steps
            print("\nSelected preprocessing steps:")
            print(f"1. Extracting target variable: 'YY_Infant_Mortality_Rate_Imr_Total_Person'")
            print(f"2. Removing columns starting with 'ZZ' and 'YY'")
            if remove_high_missing:
                print(f"3. Removing columns with >75% missing values")
                print(f"4. Handling missing values (strategy to be selected)")
            else:
                print(f"3. Handling missing values (strategy to be selected)")
            
            # Execute the preprocessor
            self.X, self.y = self.preprocessor.preprocess(remove_high_missing=remove_high_missing)
            
            # Save preprocessed data to data folder
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Save with timestamp to avoid overwriting existing files
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            X_path = os.path.join(data_dir, f'preprocessed_features_{timestamp}.csv')
            y_path = os.path.join(data_dir, f'preprocessed_target_{timestamp}.csv')
            
            self.X.to_csv(X_path, index=False)
            self.y.to_csv(y_path, index=False)
            
            print(f"\n{Fore.GREEN}✓ Preprocessed data saved to:{Style.RESET_ALL}")
            print(f"  - Features: {os.path.basename(X_path)}")
            print(f"  - Target: {os.path.basename(y_path)}")
            
            self.data_preprocessed = True
            
        except Exception as e:
            self.print_error(f"Error during preprocessing: {str(e)}")
            
        self.pause()
        
    def show_main_menu(self):
        """Display the main menu and handle user selection."""
        while True:
            self.print_header("DATA SCIENCE WORKFLOW MENU")
            
            print(f"{Fore.YELLOW}[1]{Style.RESET_ALL} Load and Explore Data")
            print(f"{Fore.YELLOW}[2]{Style.RESET_ALL} Preprocess Data")
            print(f"{Fore.YELLOW}[3]{Style.RESET_ALL} Feature Engineering")
            print(f"{Fore.YELLOW}[4]{Style.RESET_ALL} Model Selection and Training")
            print(f"{Fore.YELLOW}[5]{Style.RESET_ALL} Model Evaluation")
            print(f"{Fore.YELLOW}[6]{Style.RESET_ALL} Make Predictions")
            print(f"{Fore.YELLOW}[0]{Style.RESET_ALL} Exit")
            
            # Show data status
            print("\nStatus:")
            if self.data_loaded:
                print(f" - {Fore.GREEN}✓{Style.RESET_ALL} Data loaded")
            else:
                print(f" - {Fore.RED}✗{Style.RESET_ALL} Data not loaded")
                
            if self.data_preprocessed:
                print(f" - {Fore.GREEN}✓{Style.RESET_ALL} Data preprocessed")
            else:
                print(f" - {Fore.RED}✗{Style.RESET_ALL} Data not preprocessed")
            
            try:
                choice = input(f"\n{Fore.CYAN}Enter your choice (0-6): {Style.RESET_ALL}")
                
                if choice == '0':
                    self.print_header("EXITING APPLICATION")
                    print("Thank you for using the Data Science Workflow Tool!")
                    break
                    
                elif choice == '1':
                    self.load_data()
                    
                elif choice == '2':
                    self.preprocess_data()
                    
                elif choice == '3':
                    self.print_header("FEATURE ENGINEERING")
                    self.print_warning("Feature Engineering functionality not implemented yet.")
                    self.pause()
                    
                elif choice == '4':
                    self.print_header("MODEL SELECTION AND TRAINING")
                    self.print_warning("Model Selection and Training functionality not implemented yet.")
                    self.pause()
                    
                elif choice == '5':
                    self.print_header("MODEL EVALUATION")
                    self.print_warning("Model Evaluation functionality not implemented yet.")
                    self.pause()
                    
                elif choice == '6':
                    self.print_header("MAKE PREDICTIONS")
                    self.print_warning("Prediction functionality not implemented yet.")
                    self.pause()
                    
                else:
                    self.print_error("Invalid choice. Please enter a number between 0 and 6.")
                    self.pause()
                    
            except Exception as e:
                self.print_error(f"An error occurred: {str(e)}")
                self.pause()

# If run directly, show a message that this should be imported
if __name__ == "__main__":
    print("This module should be imported and used by main.py")
    print("Please run main.py instead.")