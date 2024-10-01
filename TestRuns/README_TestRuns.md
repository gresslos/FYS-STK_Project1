# FYS-STK_Project1 - Testrun

## Overview
Test_identity_matrix.py  :  
- Include: Starting-code for OLS, Ridge, Lasso regression on **1D Franke's Function** 
- Include: Code for OLS, Ridge, Lasso regression on **2D Franke's Function** (from bg_taskabc.py)
- Testing if MSE = 0 and R2 = 1 for OLS Regressions when **Design Matrix = Identity Matrix** in 1D and 2D 
    - **Note**: should see \( \tilde{y} = y \)   ( trained model = function for training )
    - PS: Only look at trained model, not predicted model.
- Gives plots (`FrankeFunction_1D_regression.png` and `OLS.png`) of OLS Regressions fit in respectively 1D and 2D 
- Change `test_X_is_identity_matrix = True / False` to see / not see testing with identity matrix


## How to Run the Code
```bash
$ python3 .\Test_identity_matrix.py


