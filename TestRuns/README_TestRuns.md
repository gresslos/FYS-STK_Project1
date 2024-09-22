Test_identity_matrix.py is a copy of a later version of bg_taskabc.py (small changes in setup but overall the same regression method). It include:

'test_X_is_identity_matrix = True'- statement which will allow for the code to set the design matrix to the identity matrix.

This will lead to y_tilde = y_train

(Look at y_tilde given split data to test and training data, and compara data (y_train) to fitted model (y_tilde))

The code plots MSE and R2 for y_tilde/y_train which is respectivly 0.00 and 1.00.

Plots is for design matrix = Identity Matrix 

NOTE:

Could look at the plots in TestRuns, but both the plots and task_a_b_c.py was pushed by accident. 

Do not know how to delete it from Git. 

We are new to Git.
