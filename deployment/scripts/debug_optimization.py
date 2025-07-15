#!/usr/bin/env python3
"""
Debug script to understand what's happening with the optimization results
"""
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def debug_demo_csv():
    print("🔍 Debugging DEMO.csv optimization results...")
    
    # Load DEMO.csv to check structure
    df = pd.read_csv('./data/samples/DEMO.csv')
    print(f"✓ Loaded DEMO.csv: {len(df)} rows")
    
    # Check required columns for optimization
    required_cols = ['Plant', 'Supplier', '2024 Volume (lbs)', 'DDP (USD)', 'Plant Location', 'Baseline Allocated Volume', 'Baseline Price Paid']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return
    else:
        print("✓ All required columns present")
    
    # Check baseline allocations
    baseline_data = df[df['Baseline Allocated Volume'] > 0]
    print(f"✓ Found {len(baseline_data)} rows with baseline allocations")
    
    if len(baseline_data) > 0:
        print("\n📊 Current baseline allocations:")
        for _, row in baseline_data.head(10).iterrows():
            print(f"  {row['Plant']} -> {row['Supplier']}: {row['Baseline Allocated Volume']:,.0f} lbs @ ${row['DDP (USD)']:.2f}/lb")
    
    # Check for optimization opportunities (like Charlie plant)
    charlie_data = df[df['Plant'] == 'Charlie']
    if not charlie_data.empty:
        print(f"\n🎯 Charlie plant opportunities:")
        for _, row in charlie_data.iterrows():
            status = "CURRENT" if row['Baseline Allocated Volume'] > 0 else "AVAILABLE"
            print(f"  {row['Supplier']}: ${row['DDP (USD)']:.2f}/lb ({status})")
    
    # Check suppliers and constraints
    suppliers = df['Supplier'].unique()
    print(f"\n📋 Available suppliers: {len(suppliers)}")
    for supplier in suppliers:
        supplier_data = df[df['Supplier'] == supplier]
        total_volume = supplier_data['2024 Volume (lbs)'].sum()
        print(f"  {supplier}: {total_volume:,.0f} lbs total capacity")
    
    return df

def check_mock_optimization_results(df):
    """Create mock optimization results to test drill-down logic"""
    print("\n🧪 Creating mock optimization results to test drill-down...")
    
    # Create a copy with mock optimized volumes
    results_df = df.copy()
    
    # Mock some optimization changes for testing
    # Make Charlie plant switch from Aunt Bethany to Aunt Smith
    charlie_mask = (results_df['Plant'] == 'Charlie')
    
    # Clear all optimized volumes first
    results_df['Optimized Volume'] = 0
    results_df['Optimized Price'] = 0
    results_df['Optimized Selection'] = ''
    
    # For Charlie plant, switch to Aunt Smith (should be cheaper)
    charlie_aunt_smith = ((results_df['Plant'] == 'Charlie') & 
                         (results_df['Supplier'] == 'Aunt Smith'))
    
    if charlie_aunt_smith.sum() > 0:
        # Get the volume from current baseline
        current_charlie_volume = results_df[
            (results_df['Plant'] == 'Charlie') & 
            (results_df['Baseline Allocated Volume'] > 0)
        ]['Baseline Allocated Volume'].sum()
        
        # Assign it to Aunt Smith
        results_df.loc[charlie_aunt_smith, 'Optimized Volume'] = current_charlie_volume
        results_df.loc[charlie_aunt_smith, 'Optimized Selection'] = 'X'
        
        aunt_smith_price = results_df[charlie_aunt_smith]['DDP (USD)'].iloc[0]
        results_df.loc[charlie_aunt_smith, 'Optimized Price'] = current_charlie_volume * aunt_smith_price
        
        print(f"✓ Mock: Moved Charlie plant ({current_charlie_volume:,.0f} lbs) to Aunt Smith @ ${aunt_smith_price:.2f}/lb")
    
    # Test the change detection logic
    print("\n🔍 Testing change detection logic...")
    
    # Create demand points like the actual app does
    results_df['demand_point'] = results_df['Plant'] + '_' + results_df['Product'] + '_' + results_df['Plant Location']
    
    changes_found = 0
    for demand_point in results_df['demand_point'].unique():
        demand_data = results_df[results_df['demand_point'] == demand_point]
        
        for _, row in demand_data.iterrows():
            baseline_vol = row.get('Baseline Allocated Volume', 0)
            optimized_vol = row.get('Optimized Volume', 0)
            
            if pd.isna(baseline_vol): baseline_vol = 0
            if pd.isna(optimized_vol): optimized_vol = 0
            
            if abs(baseline_vol - optimized_vol) > 0.01:
                changes_found += 1
                print(f"  Change detected: {row['Plant']} {row['Supplier']} - Baseline: {baseline_vol:,.0f}, Optimized: {optimized_vol:,.0f}")
    
    print(f"✓ Found {changes_found} allocation changes")
    
    # Find unique plants with changes
    changed_plants = set()
    for demand_point in results_df['demand_point'].unique():
        demand_data = results_df[results_df['demand_point'] == demand_point]
        
        for _, row in demand_data.iterrows():
            baseline_vol = row.get('Baseline Allocated Volume', 0)
            optimized_vol = row.get('Optimized Volume', 0)
            
            if pd.isna(baseline_vol): baseline_vol = 0
            if pd.isna(optimized_vol): optimized_vol = 0
            
            if abs(baseline_vol - optimized_vol) > 0.01:
                changed_plants.add(row['Plant'])
    
    print(f"✓ Plants with changes: {len(changed_plants)} ({list(changed_plants)})")
    
    return results_df

if __name__ == "__main__":
    df = debug_demo_csv()
    if df is not None:
        mock_results = check_mock_optimization_results(df)
        print(f"\n🎉 Drill-down should show {len(mock_results[mock_results['Optimized Volume'] > 0])} optimized allocations")