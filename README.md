# nielson_abcd_2018

This will eventually be the repo for code related to our combat paper. In the mean time, it's just the code for the simulations.

Here's the commands to create a conda env:

```
conda config --append channels bioconda
conda create -n abcd_mine python=3.6.3 jupyter notebook=5.4.1 r=3.3.1 rpy2=2.7.8 scikit-learn=0.19.1 patsy=0.4.1 pandas=0.20.3 numpy=1.13.3
. activate abcd_mine
pip install sklearn-pandas
```

