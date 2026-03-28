"""
This module contains the various procedures for processing data.
"""

import argparse
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def load_data(data_path):
    """
    Read dataset from given directory.
        Parameters:
            data_path (str): directory containing dataset in csv
        Returns:
            df: dataframe containing the input data
    """
    df = pd.read_csv(data_path)
    return df

def save_data(data_path, df):
    """
    Save data to directory.
        Parameters:
            data_path (str): Directory for saving dataset
            df: Dataframe containing data to save
        Returns:
            None: No returns required
    """
    df.to_csv(data_path.replace('.csv','_processed.csv'), index=False)
    return None

def preprocess(df):
    """
    Orchestrate data pre-processing procedures.
        Parameters:
            df: Input dataframe to be pre-processed
        Returns:
            df: Resultant dataframe after pre-processing
    """
    numeric_cols = df.columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    return df.dropna()

def run(data_path):
    """
    Main script to read and pre-process data.
        Parameters:
            data_path (str): Directory containing dataset in csv
        Returns:
            df: Dataframe containing the final pre-processed data
    """
    logging.info('Load data..')
    df = load_data(data_path)
    logging.info('Processing data...')
    df = preprocess(df)
    logging.info('Save data...')
    save_data(data_path, df)
    logging.info('Completed')
    return df

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str)
    args = argparser.parse_args()
    run(args.data_path)
