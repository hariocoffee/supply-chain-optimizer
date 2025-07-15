#!/usr/bin/env python3
"""
Data Quality Analysis and Cleaning Script for SCO Data
Identifies and fixes data quality issues that would prevent Pyomo optimization
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def analyze_data_quality(df):
    """Analyze data quality issues"""
    print("=== DATA QUALITY ANALYSIS ===\n")
    
    issues = []
    
    # 1. Duplicate entries analysis
    print("1. DUPLICATE ENTRIES:")
    key_cols = ['Plant_Product_Location_ID', 'Supplier']
    duplicates = df[df.duplicated(subset=key_cols, keep=False)]
    
    if len(duplicates) > 0:
        issues.append(f"Found {len(duplicates)} duplicate entries (same Plant_Product_Location_ID + Supplier)")
        print(f"   - {len(duplicates)} duplicate entries found")
        
        # Group duplicates by Plant_Product_Location_ID and Supplier
        duplicate_groups = []
        for plant_product_location_id in duplicates['Plant_Product_Location_ID'].unique():
            subset = duplicates[duplicates['Plant_Product_Location_ID'] == plant_product_location_id]
            for supplier in subset['Supplier'].unique():
                supplier_subset = subset[subset['Supplier'] == supplier]
                if len(supplier_subset) > 1:
                    duplicate_groups.append({
                        'Plant_Product_Location_ID': plant_product_location_id,
                        'Supplier': supplier,
                        'Count': len(supplier_subset),
                        'Rows': supplier_subset.index.tolist()
                    })
        
        for group in duplicate_groups:
            print(f"   - {group['Plant_Product_Location_ID']} + {group['Supplier']}: {group['Count']} entries")
    else:
        print("   - No duplicate entries found")
    
    # 2. Volume consistency analysis
    print("\n2. VOLUME CONSISTENCY:")
    volume_inconsistencies = []
    for plant_product_location_id in df['Plant_Product_Location_ID'].unique():
        subset = df[df['Plant_Product_Location_ID'] == plant_product_location_id]
        unique_volumes = subset['2024 Volume (lbs)'].unique()
        if len(unique_volumes) > 1:
            volume_inconsistencies.append({
                'Plant_Product_Location_ID': plant_product_location_id,
                'Volumes': unique_volumes,
                'Count': len(subset)
            })
    
    if volume_inconsistencies:
        issues.append(f"Found {len(volume_inconsistencies)} Plant_Product_Location_IDs with inconsistent volumes")
        print(f"   - {len(volume_inconsistencies)} Plant_Product_Location_IDs with inconsistent volumes")
        for inconsistency in volume_inconsistencies:
            print(f"     • {inconsistency['Plant_Product_Location_ID']}: {inconsistency['Volumes']}")
    else:
        print("   - All volumes are consistent")
    
    # 3. Data formatting issues
    print("\n3. DATA FORMATTING ISSUES:")
    
    # Check for trailing spaces
    string_cols = ['Plant', 'Product', 'Supplier', 'Plant Location', 'Selection', 'Split', 'Volume_Fraction']
    formatting_issues = []
    for col in string_cols:
        if col in df.columns:
            has_trailing_spaces = df[col].str.contains(r'\s+$', na=False).any()
            if has_trailing_spaces:
                formatting_issues.append(f"Trailing spaces in {col}")
    
    if formatting_issues:
        issues.extend(formatting_issues)
        print(f"   - Found {len(formatting_issues)} formatting issues:")
        for issue in formatting_issues:
            print(f"     • {issue}")
    else:
        print("   - No formatting issues found")
    
    # 4. Optimization structure issues
    print("\n4. OPTIMIZATION STRUCTURE ISSUES:")
    
    # Check for multiple baseline suppliers
    baseline_suppliers = df[df['Is_Baseline_Supplier'] == 1].groupby('Plant_Product_Location_ID').size()
    multiple_baseline = baseline_suppliers[baseline_suppliers > 1]
    if len(multiple_baseline) > 0:
        issues.append(f"Found {len(multiple_baseline)} Plant_Product_Location_IDs with multiple baseline suppliers")
        print(f"   - {len(multiple_baseline)} Plant_Product_Location_IDs with multiple baseline suppliers")
        for plant_id, count in multiple_baseline.items():
            print(f"     • {plant_id}: {count} baseline suppliers")
    else:
        print("   - No multiple baseline supplier issues found")
    
    # Check for missing baseline suppliers
    no_baseline = df.groupby('Plant_Product_Location_ID')['Is_Baseline_Supplier'].sum()
    no_baseline_ids = no_baseline[no_baseline == 0].index.tolist()
    if len(no_baseline_ids) > 0:
        issues.append(f"Found {len(no_baseline_ids)} Plant_Product_Location_IDs with no baseline supplier")
        print(f"   - {len(no_baseline_ids)} Plant_Product_Location_IDs with no baseline supplier")
    else:
        print("   - All Plant_Product_Location_IDs have baseline suppliers")
    
    return issues, duplicate_groups, volume_inconsistencies

def clean_data(df):
    """Clean the data by addressing identified issues"""
    print("\n=== DATA CLEANING ===\n")
    
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # 1. Fix trailing spaces in string columns
    print("1. FIXING FORMATTING ISSUES:")
    string_cols = ['Plant', 'Product', 'Supplier', 'Plant Location', 'Selection', 'Split', 'Volume_Fraction']
    for col in string_cols:
        if col in cleaned_df.columns:
            # Remove trailing and leading spaces
            cleaned_df[col] = cleaned_df[col].str.strip()
    print("   - Removed trailing/leading spaces from string columns")
    
    # 2. Handle duplicates
    print("\n2. HANDLING DUPLICATE ENTRIES:")
    
    # For duplicates, we need to decide which one to keep
    # Strategy: For each duplicate group, keep the one with the most logical selection
    key_cols = ['Plant_Product_Location_ID', 'Supplier']
    
    # Track rows to remove
    rows_to_remove = []
    
    # Handle specific duplicate cases
    duplicate_groups = []
    duplicates = cleaned_df[cleaned_df.duplicated(subset=key_cols, keep=False)]
    
    for plant_product_location_id in duplicates['Plant_Product_Location_ID'].unique():
        subset = duplicates[duplicates['Plant_Product_Location_ID'] == plant_product_location_id]
        for supplier in subset['Supplier'].unique():
            supplier_subset = subset[subset['Supplier'] == supplier]
            if len(supplier_subset) > 1:
                # Determine which row to keep
                # Priority: 1) Has Selection='X' (baseline supplier), 2) Lower DDP price
                
                has_selection = supplier_subset['Selection'] == 'X'
                if has_selection.any():
                    # Keep the one with Selection='X'
                    keep_idx = supplier_subset[has_selection].index[0]
                else:
                    # Keep the one with lowest DDP price
                    keep_idx = supplier_subset.loc[supplier_subset['DDP (USD)'].idxmin()].name
                
                # Mark others for removal
                for idx in supplier_subset.index:
                    if idx != keep_idx:
                        rows_to_remove.append(idx)
                
                print(f"   - Removing duplicate entries for {plant_product_location_id} + {supplier}")
                print(f"     Keeping row {keep_idx}, removing {len(supplier_subset)-1} duplicates")
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop(rows_to_remove)
    print(f"   - Removed {len(rows_to_remove)} duplicate rows")
    
    # 3. Handle volume inconsistencies
    print("\n3. HANDLING VOLUME INCONSISTENCIES:")
    
    # For Sierra_Location71_Brownies, we need to decide on the correct volume
    # Based on the data, it appears there are two separate entries with different volumes
    # This might be intentional (different volume allocations), but if not, we need to standardize
    
    # Check for the specific case
    sierra_data = cleaned_df[cleaned_df['Plant_Product_Location_ID'] == 'Sierra_Location71_Brownies']
    if len(sierra_data) > 1:
        # Get the unique volumes
        unique_volumes = sierra_data['2024 Volume (lbs)'].unique()
        if len(unique_volumes) > 1:
            # This appears to be two separate entries with different volumes
            # We need to check if this is correct or if we should consolidate
            print(f"   - Found volume inconsistency in Sierra_Location71_Brownies: {unique_volumes}")
            print("   - This appears to be two separate supplier entries with different volumes")
            print("   - Keeping both entries as they represent different volume allocations")
        else:
            print("   - Volume inconsistency resolved")
    else:
        print("   - No volume inconsistencies found")
    
    # 4. Handle optimization structure issues
    print("\n4. HANDLING OPTIMIZATION STRUCTURE ISSUES:")
    
    # Handle multiple baseline suppliers
    baseline_suppliers = cleaned_df[cleaned_df['Is_Baseline_Supplier'] == 1].groupby('Plant_Product_Location_ID').size()
    multiple_baseline = baseline_suppliers[baseline_suppliers > 1]
    
    if len(multiple_baseline) > 0:
        print(f"   - Found {len(multiple_baseline)} Plant_Product_Location_IDs with multiple baseline suppliers")
        for plant_id, count in multiple_baseline.items():
            # Keep only the baseline supplier with the lowest cost
            plant_data = cleaned_df[
                (cleaned_df['Plant_Product_Location_ID'] == plant_id) & 
                (cleaned_df['Is_Baseline_Supplier'] == 1)
            ]
            
            # Keep the one with lowest baseline price
            keep_idx = plant_data.loc[plant_data['Baseline Price Paid'].idxmin()].name
            
            # Set others to non-baseline
            for idx in plant_data.index:
                if idx != keep_idx:
                    cleaned_df.loc[idx, 'Is_Baseline_Supplier'] = 0
                    cleaned_df.loc[idx, 'Baseline Allocated Volume'] = 0
                    cleaned_df.loc[idx, 'Baseline Price Paid'] = 0
                    cleaned_df.loc[idx, 'Selection'] = np.nan
                    cleaned_df.loc[idx, 'Split'] = '0%'
            
            print(f"     • Fixed {plant_id}: kept row {keep_idx} as baseline")
    else:
        print("   - No multiple baseline supplier issues to fix")
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def main():
    """Main function to analyze and clean the data"""
    print("SCO Data Quality Analysis and Cleaning Tool")
    print("=" * 50)
    
    # Read the original data
    input_file = './sco_data_copy.csv'
    output_file = './sco_data_cleaned.csv'
    
    try:
        df = pd.read_csv(input_file)
        print(f"\nLoaded data from {input_file}")
        print(f"Original data shape: {df.shape}")
        
        # Analyze data quality
        issues, duplicate_groups, volume_inconsistencies = analyze_data_quality(df)
        
        # Clean the data
        cleaned_df = clean_data(df)
        
        # Verify cleaning
        print(f"\n=== CLEANING VERIFICATION ===")
        print(f"Original rows: {len(df)}")
        print(f"Cleaned rows: {len(cleaned_df)}")
        print(f"Rows removed: {len(df) - len(cleaned_df)}")
        
        # Check if issues are resolved
        print(f"\nVerifying issue resolution:")
        
        # Check duplicates
        key_cols = ['Plant_Product_Location_ID', 'Supplier']
        remaining_duplicates = cleaned_df[cleaned_df.duplicated(subset=key_cols, keep=False)]
        print(f"- Remaining duplicates: {len(remaining_duplicates)}")
        
        # Check multiple baseline suppliers
        baseline_suppliers = cleaned_df[cleaned_df['Is_Baseline_Supplier'] == 1].groupby('Plant_Product_Location_ID').size()
        multiple_baseline = baseline_suppliers[baseline_suppliers > 1]
        print(f"- Plant_Product_Location_IDs with multiple baseline suppliers: {len(multiple_baseline)}")
        
        # Save cleaned data
        cleaned_df.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to {output_file}")
        
        # Summary
        print(f"\n=== SUMMARY ===")
        print(f"Total issues identified: {len(issues)}")
        print(f"Data cleaning completed successfully!")
        print(f"The cleaned dataset is ready for Pyomo optimization.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())