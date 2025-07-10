"""
Script:          correlation_utils.py
Purpose:         Provides utility functions for correlation analysis
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            07-07-2025

PyTorch Version: 2.7.1
"""
import pandas as pd
import glob
import os
import re

def find_common_stains(panel_one, panel_two):
    # Returns the common set of stains in the two panels
    return list(set(panel_one) & set(panel_two))

def load_ws_mapping(ws_mapping_path, ws_dir):
    # Loads the WS-IMC mapping as a pandas dataframe: Barcode to CaseID

    ## Load the mapping
    ws_mapping = pd.read_excel(ws_mapping_path)

    ## Extract the relevant columns and format them
    ws_mapping = ws_mapping[['Barcode or UHNL_ID', 'CaseID']]
    ws_mapping.columns = ['Barcode', 'CaseID']
    ws_mapping['Barcode'] = ws_mapping['Barcode'].apply(
        lambda s: re.sub(r'^(\d+)\s*[Tt]\s*([0-9]+)[- ]?([0-9]*)', 
                         lambda m: f"{m.group(1)}t{m.group(2)}{m.group(3)}", str(s))
    )

    ## Fetch the available WS images via barcodes
    ws_imgs = glob.glob(os.path.join(ws_dir, "*.mcd"))
    ws_imgs = [os.path.basename(f) for f in ws_imgs]
    ws_barcodes = [f.split('_')[2] for f in ws_imgs]

    ## Exclude mapping entries with data unavailable
    ws_mapping = ws_mapping[ws_mapping['Barcode'].isin(ws_barcodes)]

    return ws_mapping

def load_tma_mapping(tma_mapping_path, ws_mapping, tma_dir):
    # Loads the TMA-IMC mapping as a pandas dataframe: TiffName to CaseID

    ## Load the mappings
    tma_mapping = pd.read_csv(tma_mapping_path)
    
    ## Extract the relevant columns and format them
    tma_mapping = tma_mapping[['TiffName', 'Sample']]
    tma_mapping = tma_mapping[tma_mapping['Sample'].str.contains('Case', na = False)]
    tma_mapping['Sample'] = tma_mapping['Sample'].str.extract(r'(Case\d+)', expand=False)
    tma_mapping.columns = ['TiffName', 'CaseID']

    ## Fetch the image paths
    tma_imgs = glob.glob(os.path.join(tma_dir, "*.tiff"))
    tma_imgs = [os.path.basename(f) for f in tma_imgs]

    ## Exclude mapping entries with data unavailable
    tma_mapping = tma_mapping[tma_mapping['TiffName'].isin(tma_imgs)]

    ## Exclude TMAs that don't map to a WS image
    tma_mapping = tma_mapping[tma_mapping['CaseID'].isin(ws_mapping['CaseID'])]

    return tma_mapping



def load_mapping(ws_mapping_path, tma_mapping_dir, ws_dir, tma_dir):
    # Loads the WS to TMA mapping as a dictionary: barcode to caseID

    ## Load the mappings
    ws_mapping = pd.read_excel(ws_mapping_path)
    tma_mapping = consolidate_mappings(tma_mapping_dir)
    
    ## Extract the relevant columns and format them

    # Map Barcode to CaseID, reformatting any alphabetical characters
    ws_mapping = ws_mapping[['Barcode or UHNL_ID', 'CaseID']]
    ws_mapping.columns = ['Barcode', 'CaseID']
    ws_mapping['Barcode'] = ws_mapping['Barcode'].apply(
        lambda s: re.sub(r'^(\d+)\s*[Tt]\s*([0-9]+)[- ]?([0-9]*)', 
                         lambda m: f"{m.group(1)}t{m.group(2)}{m.group(3)}", str(s))
    )

    # Map Tiff to CaseID, excluding any that don't contain CaseID
    tma_mapping = tma_mapping[['TiffName', 'Sample']]
    tma_mapping = tma_mapping[tma_mapping['Sample'].str.contains('Case', na = False)]
    tma_mapping['Sample'] = tma_mapping['Sample'].str.extract(r'(Case\d+)', expand=False)
    tma_mapping.columns = ['TiffName', 'CaseID']

    ## Fetch the image paths

    # Fetch the available WS images via barcodes
    ws_imgs = glob.glob(os.path.join(ws_dir, "*.mcd"))
    ws_imgs = [os.path.basename(f) for f in ws_imgs]
    ws_barcodes = [f.split('_')[2] for f in ws_imgs]
    
    # Fetch the available TMA images via filename
    tma_imgs = glob.glob(os.path.join(tma_dir, "*.tiff"))
    tma_imgs = [os.path.basename(f) for f in tma_imgs]
    
    ## Exclude mapping entries with data unavailable
    ws_mapping = ws_mapping[ws_mapping['Barcode'].isin(ws_barcodes)]
    tma_mapping = tma_mapping[tma_mapping['TiffName'].isin(tma_imgs)]
    
    ## Exclude TMAs that don't map to a WS image
    tma_mapping = tma_mapping[tma_mapping['CaseID'].isin(ws_mapping['CaseID'])]

    return ws_mapping, tma_mapping

def extract_tma_id(img_path):
    # Extracts the PDAC number and CaseID if it exists
    filename = os.path.basename(img_path)
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    return (parts[0], parts[3]) if len(parts) >= 4 else None
    
def extract_ws_id(img_path):
    # Extracts the Barcode from the WS-IMC image path
    filename = os.path.basename(img_path)
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    return parts[2]
    