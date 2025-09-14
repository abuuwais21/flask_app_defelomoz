# -*- coding: utf-8 -*-
"""
Data Processing Module for Waste Management Time Series Analysis
Handles CSV data loading, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class DataProcessor:
    """Class for processing waste management data"""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
    
    def load_and_process_data(self, csv_path):
        """
        Load and process CSV data
        
        Args:
            csv_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        try:
            # Load the CSV data
            df = pd.read_csv(csv_path)
            
            # Clean column names (remove extra spaces)
            df = df.rename(columns=lambda x: x.strip())
            
            # Convert Timestamp to datetime
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            
            # Clean the data
            df_clean = df.copy()
            
            # Handle missing values in Jumlah column
            df["Jumlah"] = df["Jumlah"].fillna(1)
            df["Jumlah"] = pd.to_numeric(df["Jumlah"], errors="coerce").fillna(1).astype(float)
            
            # Handle missing values in Kategori column
            df["Kategori"] = df["Kategori"].fillna("Unknown")
            
            # Store processed data
            self.data = df
            self.processed_data = df.copy()
            
            return df
            
        except Exception as e:
            raise Exception(f"Error processing data: {str(e)}")
    
    def get_data_info(self):
        """
        Get information about the loaded data
        
        Returns:
            dict: Data information
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        return {
            "shape": self.data.shape,
            "columns": self.data.columns.tolist(),
            "data_types": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "date_range": {
                "start": self.data["Timestamp"].min().strftime('%Y-%m-%d %H:%M:%S'),
                "end": self.data["Timestamp"].max().strftime('%Y-%m-%d %H:%M:%S')
            },
            "unique_places": self.data["Tempat"].nunique(),
            "unique_categories": self.data["Kategori"].nunique()
        }
    
    def get_places(self):
        """
        Get list of unique places
        
        Returns:
            list: List of unique places
        """
        if self.data is None:
            return []
        
        return self.data["Tempat"].unique().tolist()
    
    def get_place_data(self, place):
        """
        Get data for a specific place
        
        Args:
            place (str): Name of the place
            
        Returns:
            pd.DataFrame: Data for the specified place
        """
        if self.data is None:
            return pd.DataFrame()
        
        return self.data[self.data["Tempat"] == place].copy()
    
    def get_summary_stats(self):
        """
        Get summary statistics for the data
        
        Returns:
            dict: Summary statistics
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        return {
            "total_records": len(self.data),
            "waste_stats": {
                "mean": self.data["Jumlah"].mean(),
                "median": self.data["Jumlah"].median(),
                "std": self.data["Jumlah"].std(),
                "min": self.data["Jumlah"].min(),
                "max": self.data["Jumlah"].max()
            },
            "places_stats": self.data.groupby("Tempat")["Jumlah"].agg(['count', 'mean', 'sum']).to_dict('index'),
            "category_stats": self.data.groupby("Kategori")["Jumlah"].agg(['count', 'mean', 'sum']).to_dict('index')
        }
    
    def validate_data(self):
        """
        Validate the loaded data
        
        Returns:
            dict: Validation results
        """
        if self.data is None:
            return {"valid": False, "error": "No data loaded"}
        
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check for required columns
        required_columns = ["Timestamp", "Tempat", "Jumlah", "Kategori"]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Missing required columns: {missing_columns}")
        
        # Check for empty dataset
        if len(self.data) == 0:
            validation_results["valid"] = False
            validation_results["errors"].append("Dataset is empty")
        
        # Check for duplicate timestamps
        duplicate_timestamps = self.data.duplicated(subset=["Timestamp", "Tempat"]).sum()
        if duplicate_timestamps > 0:
            validation_results["warnings"].append(f"Found {duplicate_timestamps} duplicate timestamp-place combinations")
        
        # Check for negative waste amounts
        negative_amounts = (self.data["Jumlah"] < 0).sum()
        if negative_amounts > 0:
            validation_results["warnings"].append(f"Found {negative_amounts} records with negative waste amounts")
        
        return validation_results
