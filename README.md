# FYS-STK_Project1

## Overview
- bg_taskabc.py  :   OLS, Ridge, Lasso regression on **Franke's Function**   
- task_g.py      :   OLS, Ridge, Lasso regression on **real topografic data** from Lausanne (dataset: `Lausanne.tif`)
  
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
