# nielson_abcd_2018

This will eventually be the repo for code related to our combat paper. In the mean time, it's just the code for the simulations.

To run the notebooks (using docker) clone this repository and cd into it on the command line then type the following with the appropriate absolute path to your data directory:

```
docker run -it -v [data_dir_absolute_path]:/data -v $PWD:/mnt --user root -e NB_GID=`id -u` -e NB_UID=`id -u` --rm -p 8888:8888 nihfmrif/abcd_combo start.sh jupyter lab
```

Now just paste the url listed in your terminal into your browser. You will only have access to the directory from which you ran the docker command and your data directory (located at /data).

### Deprecated:
Here's the commands to create a conda env:

```
conda config --append channels bioconda
conda create -n abcd_mine python=3.6.3 jupyter notebook=5.4.1 r=3.3.1 rpy2=2.7.8 scikit-learn=0.19.1 patsy=0.4.1 pandas=0.20.3 numpy=1.13.3
. activate abcd_mine
pip install sklearn-pandas
```

