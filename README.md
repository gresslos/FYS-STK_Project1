# FYS-STK_Project1

## Overview
- bg_taskabc.py  :   OLS, Ridge, Lasso regression on **Franke's Function**
- op_task_e.py   :   Bootstrap resampling analysis for number of points and complexity on **Franke's Function** with bias variance trade-off
- op_task_f.py   :   K-fold cross validation resampling analysis for complexity on **Franke's Function**.
- task_g.py      :   OLS, Ridge, Lasso regression on **real topografic data** from Lausanne (dataset: `Lausanne.tif`), then cross validation to validate the optimal complexity of the model
- Additional_Plots folder   : Include plots in report and additional plots
- TestRuns folder           : Include testing of bg_taskabc.py with **Design Matrix = Identity Matrix**
- Surfaces                  : .tif file with **Lausanne Topographic Data**
  


## Functions
### `Regression-Function` (found in `bg_taskabc.py`)
- **Input Variables**:
  - `x`, `y`, `z`: Positions of the surface.
  - `max_degree`: Maximum polynomial degree of the model.
  - `want_beta`: Boolean flag to print out \( \lambda \), MSE, R2, and \( \beta \) (regression coefficients).  
    - **Note**: To obtain the \( \beta \)-values, set `want_beta = True` (line 437).
- **Structure**: detailed structure provided in `Project1.tex`.

### `Bootstrap (number of points)` (found in `op_task_e.py`)
- **Input Variables**:
  - `data`: Directory path to topographic data, ex: 'Surfaces/Lausanna.tif'.
  - `use_real_data`: Boolean value to decide whether to use real data or constructed data.
  - `n_bootstraps`: Number of times to do bootstrap resampling.
  - `row_start`: Decide the index for the datapoint (along one axis) where you want to start.
  - `special_deg`: Specific degree for the model that you want to analyze
  - `min_n`: Minimum number of datapoints you want to use for the analysis.
  - `max_n`: Maximum number of datapoints you want to use for the analysis.
  - `interval`: Decide the interval between the minimum and maximum number of datapoints.
- **Note**: The function was modified to be able to be used on real data, but it was never used for that purpose because of time constraint.

### `Bootstrap (complexity)` (found in `op_task_e.py`)
  **Input Variables**:
  - `x`, `y`, `z`: Positions of the surface.
  - `n_bootstraps`: Number of times to do bootstrap resampling.
  - `min_deg`: Minimum degree of complexity you want to use for the analysis.
  - `max_deg`: Maximum degree of complexity you want to use for the analysis.
  - `interval`: Decide the interval between the minimum and maximum degree of complexity.

### `Cross Validation` (found in `op_task_f.py`)
  **Input Variables**:
  - `x`, `y`, `z`: Positions of the surface.
  - `min_deg`: Minimum degree of complexity you want to use for the analysis.
  - `max_deg`: Maximum degree of complexity you want to use for the analysis.
  - `interval`: Decide the interval between the minimum and maximum degree of complexity.
  - `k`: Number of folds for the k-fold cross validation scheme
  - `a`: The smallest tested lambda value is 10^a
  - `b`: The largest tested lambda value is 10^b

### `Cross Validation without Lasso` (found in `op_task_f.py`)
  **Input Variables**:
  - `x`, `y`, `z`: Positions of the surface.
  - `min_deg`: Minimum degree of complexity you want to use for the analysis.
  - `max_deg`: Maximum degree of complexity you want to use for the analysis.
  - `interval`: Decide the interval between the minimum and maximum degree of complexity.
  - `k`: Number of folds for the k-fold cross validation scheme
  - `a`: The smallest tested lambda value is 10^a
  - `b`: The largest tested lambda value is 10^b
- **Note**: This function was made to test the cross validation function for large complexities since the Lasso part of the function contributed the most to the computation time. There were some issues with convergence.




## How to Run the Code
```bash
$ nohup python3 bg_taskabc.py > output_test.log 2>&1 &
$ nohup python3 op_task_e.py > output_boot.log 2>&1 &
$ nohup python3 op_task_f.py > output_cv.log 2>&1 &
$ nohup python3 task_g.py > output_real.log 2>&1 &