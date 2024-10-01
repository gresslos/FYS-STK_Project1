# FYS-STK_Project1

## Overview
- bg_taskabc.py  :   OLS, Ridge, Lasso regression on **Franke's Function**
- YOUR .py OLP
- task_g.py      :   OLS, Ridge, Lasso regression (LEGG TIL KFOLD OLE PETTER)  on **real topografic data** from Lausanne (dataset: `Lausanne.tif`)
- Additional_Plots folder   : Include plots in report and additional plots
- TestRuns folder           : Include testing of bg_taskabc.py with **Design Matrix = Identity Matrix**
  
## Functions
### `Regression-Function` (found in `bg_taskabc.py`)
- **Input Variables**:
  - `x`, `y`, `z`: Positions of the surface.
  - `max_degree`: Maximum polynomial degree of the model.
  - `want_beta`: Boolean flag to print out \( \lambda \), MSE, R2, and \( \beta \) (regression coefficients).  
    - **Note**: To obtain the \( \beta \)-values, set `want_beta = True` (line 437).
- **Structure**: detailed structure provided in `Project1.tex`.

     
  Ole Petter: putt in some important functions structure



## How to Run the Code
```bash
$ nohup python3 bg_taskabc.py > output_test.log 2>&1 &
$ nohup python3 task_g.py > output_real.log 2>&1 &
