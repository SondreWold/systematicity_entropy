# Data files

# Experiment 1
These files were created by first creating all possible trees from the CFG.
In this complete data, we then sampled 6000 instances of the dataset where
jump always appears in the one embedded sentence for "and" and in the
other embedded sentence for "after". For this experiment, the restricted
verb is "jump". 

Consequently, the test set always contains:

- jump and x
- x after jump

While the training set never contains these combinations. 

This first dataset is the degenerate case (H=0).
The low, medium, high prefixes refers to the number of samples in the
training set, where low=3000, medium=4000, and high=6000. These are used
for the ablation experiment on the effect of unique sample size.

In order to increase H, we subtracted a number of samples with (jump after
x) and (x and jump) from the total sample population and added that same
amount evenly for all the other verbs. This mixes the degenerate and the
uniform. We did this for eight values of 8 (every 0.5, getting as close as
we could get). The test set remains the same for all settings.

# Experiment 2
These files were created directly from the CFG in `./grammars`. A
description of the generation procedure can be found there. 

In this experiment, the restricted verb is "lunge". 

The test set always have:
- x and lunge
- lunge after x

While the training set never contains these combinations. 
