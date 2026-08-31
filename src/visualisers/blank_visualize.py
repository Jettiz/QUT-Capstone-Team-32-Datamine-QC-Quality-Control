'''
This will be the visualisation model for the blank sample types.
'''

# build inputs -> outputs
# create base operations that we can use to specialisations.
# focus on making base objects.

# reccomendation create epic on jira for visualiser object -- create shell of object.
# making PR, to make them small.
# next task -> create new branch!
#


from matplotlib import colors
import pandas as pd
import numpy as np
from .base_detector import LCSAnomalyResult
from typing import Dict, Any, Tuple, Optional
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

from BLANK_loader import BlankLoader

BlankLoader()

# first object to show how many potential missing values there are

def BLANK_MissingValues(df: pd.DataFrame):
    m_values = df.isnull().sum()
    sortd_m_values = m_values[m_values > 0].sort_values(ascending=False)
    plt.bar(data=sortd_m_values, x=sortd_m_values.index, height=sortd_m_values.values)
    plt.title('Missing Values')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# Make a graph showing all the final results of blank samples compared to their target samples

def BLANK_FinalResults():
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot() # Final Values
    sns.scatterplot() # Target Value to compare
    colors = ['blue' if STANDARD_STATUS == 'Pass' else 'red' if STANDARD_STATUS == 'UpperFailure' else 'red' if STANDARD_STATUS == 'LowerFailure' else 'yellow' if STANDARD_STATUS == 'UpperWarning' or 'LowerWarning' else 'orange' if STANDARD_STATUS == 'IgnoredUpperFailure' or 'IgnoredLowerFailure']
    plt.xlabel("Sample")
    plt.ylabel("Value")

def BLANK_FinalValues(df pd.DataFrame):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='SAMPLE_ID', y='NUMERIC_FINAL_VALUE', hue='STANDARD_STATUS', palette=colors)
    plt.title('Final Values of Blank Samples')
    plt.xlabel('Sample ID')
    plt.ylabel('Final Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


from blank_detector.py import BlankDriftDetector

def ShowBlankDrift():
    BlankDriftDetector()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=blank_historicalDrift, x='ANALYSED_DATE', y='NUMERIC_FINAL_VALUE')
    plt.title('Drift of Blank Samples Over Time')
    plt.xlabel('Date')
    plt.ylabel('Final Value')
    plt.show()

# show only failures

 def BlankFailures():
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df[df['STANDARD_STATUS'].isin(['UpperFailure', 'LowerFailure'])], x='SAMPLE_ID', y='NUMERIC_FINAL_VALUE', hue='STANDARD_STATUS', palette=colors)
    plt.title('Failures of Blank Samples')
    plt.xlabel('Sample ID')
    plt.ylabel('Final Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


