# Data Quality Analysis Report
## SCO Data Cleaning for Pyomo Optimization

### Executive Summary

This report documents the data quality issues identified in the SCO (Supply Chain Optimization) dataset and the cleaning procedures applied to prepare the data for Pyomo optimization. The original dataset contained 190 rows and 24 columns, with several critical issues that would have prevented successful optimization.

### Data Quality Issues Identified

#### 1. Duplicate Entries (Critical Issue)
- **Problem**: 22 duplicate entries with identical Plant_Product_Location_ID + Supplier combinations
- **Impact**: Would cause optimization model conflicts and incorrect constraint definitions
- **Affected Records**: 11 unique Plant_Product_Location_ID + Supplier combinations with 2 entries each

**Specific Duplicates Found:**
- Golf_Location7_Brownies + Aunt Smith (2 entries)
- Juliet_Location10_Brownies + Aunt Smith (2 entries) 
- Kilo_Location11_Brownies + Aunt Smith (2 entries)
- Mike_Location13_Brownies + Aunt Smith (2 entries)
- November_Location14_Brownies + Aunt Smith (2 entries)
- Papa_Location16_Brownies + Aunt Smith (2 entries)
- Quebec_Location17_Brownies + Aunt Smith (2 entries)
- Uniform_Location47_Brownies + Aunt Bethany (2 entries)
- Zulu_Location52_Brownies + Aunt Bethany (2 entries)
- Echo_Location57_Brownies + Aunt Bethany (2 entries)
- Sierra_Location71_Brownies + Aunt Smith (2 entries)

#### 2. Volume Inconsistencies (Critical Issue)
- **Problem**: 1 Plant_Product_Location_ID with inconsistent volume values
- **Impact**: Would create impossible optimization constraints
- **Affected Record**: Sierra_Location71_Brownies with volumes [8,600,000, 12,300,000]

#### 3. Data Formatting Issues (Minor Issue)
- **Problem**: Trailing spaces in Volume_Fraction column
- **Impact**: Could cause string matching issues in optimization code
- **Affected Column**: Volume_Fraction contained trailing spaces

#### 4. Optimization Structure Issues (Critical Issue)
- **Problem**: 1 Plant_Product_Location_ID with multiple baseline suppliers
- **Impact**: Would violate optimization constraint that each location can have only one baseline supplier
- **Affected Record**: Sierra_Location71_Brownies had 2 baseline suppliers

### Cleaning Procedures Applied

#### 1. Duplicate Removal Strategy
For each duplicate group, the following priority was applied:
1. **Primary Priority**: Keep entry with Selection='X' (baseline supplier)
2. **Secondary Priority**: Keep entry with lowest DDP (USD) price

**Results**: 11 duplicate rows removed, retaining the most logical entry for each combination.

#### 2. Volume Consistency Resolution
- Analyzed volume inconsistencies in Sierra_Location71_Brownies
- Determined that different volumes represented legitimate separate supplier allocations
- No volume standardization was required after duplicate removal

#### 3. Data Formatting Fixes
- Removed leading and trailing spaces from all string columns
- Standardized Volume_Fraction formatting
- Ensured consistent data types across all columns

#### 4. Optimization Structure Fixes
- Resolved multiple baseline supplier issues through duplicate removal
- Verified that each Plant_Product_Location_ID has exactly one baseline supplier
- Ensured data structure supports proper optimization constraints

### Cleaned Dataset Summary

#### Before Cleaning:
- **Rows**: 190
- **Columns**: 24
- **Duplicates**: 22
- **Volume Inconsistencies**: 1
- **Multiple Baseline Suppliers**: 1

#### After Cleaning:
- **Rows**: 179
- **Columns**: 24
- **Duplicates**: 0
- **Volume Inconsistencies**: 0
- **Multiple Baseline Suppliers**: 0

#### Data Structure for Optimization:
- **Plant_Product_Location_IDs**: 97 unique locations
- **Suppliers**: 4 unique suppliers (Aunt Smith, Aunt Bethany, Aunt Baker, Aunt Celine)
- **Baseline Suppliers**: 97 (one per location)
- **Alternative Suppliers**: 82 (additional options for optimization)

### Key Metrics

| Metric | Value |
|--------|-------|
| Total Volume (lbs) | 12.5 billion |
| Average DDP Price | $7.37 |
| Price Range | $5.01 - $9.97 |
| Largest Volume Location | Uniform_Location47_Brownies (507.8M lbs) |
| Smallest Volume Location | November_Location14_Brownies (0.6M lbs) |

### Optimization Readiness

The cleaned dataset is now ready for Pyomo optimization with the following characteristics:

1. **Unique Constraints**: Each Plant_Product_Location_ID + Supplier combination is unique
2. **Consistent Volumes**: All volume data is consistent within each location
3. **Proper Baseline Structure**: Each location has exactly one baseline supplier
4. **Complete Data**: All required fields for optimization are populated
5. **Standardized Format**: All string fields are properly formatted

### Recommendations

1. **Data Validation**: Implement automated data validation checks before future optimization runs
2. **Duplicate Prevention**: Add unique constraints to data collection processes
3. **Volume Verification**: Implement volume consistency checks across supplier entries
4. **Baseline Supplier Rules**: Enforce single baseline supplier constraint in data entry

### Files Generated

1. **sco_data_cleaned.csv**: The cleaned dataset ready for optimization
2. **data_cleaning_script.py**: Python script for reproducible data cleaning
3. **data_quality_report.md**: This comprehensive analysis report

### Technical Details

The cleaning process was implemented using Python pandas with the following approach:
- Systematic duplicate identification and intelligent removal
- Volume consistency analysis and resolution
- String formatting standardization
- Optimization structure validation
- Comprehensive verification of results

The cleaned dataset maintains all original data integrity while resolving issues that would prevent successful Pyomo optimization.