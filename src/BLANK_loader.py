


import pandas as pd
import numpy as np
from .base_detector import LCSAnomalyResult
from typing import Dict, Any, Tuple, Optional
import tempfile


def BlankLoader():

    df = pd.read_csv(<imported data>) # import the raw sample data

    # cleaning and filling blank values to ensure the values are correct 

    blankdf = df.loc[df['ANALYTICAL_TYPE'] == 'Blank']

    blankdf['INTERNAL_MIN_VALUE'].fillna(value=-0.25, inplace=True)
    blankdf['INTERNAL_MAX_VALUE'].fillna(value=0.25, inplace=True)
    blankdf['INTERNAL_MIN_INCLUSIVE'].fillna(value='Y', inplace=True)
    blankdf['INTERNAL_MAX_INCLUSIVE'].fillna(value='Y', inplace=True)

    BLANK_REQCOL = ["NUMERIC_FINAL_VALUE", "ANALYSED_DATE", "ANALYTE_CODE", "UNIT_CODE"]
    blankdf["TARGET_VALUE"] = pd.to_numeric(blankdf["TARGET_VALUE"], errors='coerce')
    analysis_df = blankdf.dropna(subset=BLANK_REQCOL).copy()


    df = analysis_df


