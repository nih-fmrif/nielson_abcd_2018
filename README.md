# Detecting and harmonizing scanner differences in the ABCD study - annual release 2.0

#### BioArXiv Preprint: [https://doi.org/10.1101/309260](http://dx.doi.org/10.1101/309260)

The code here is not 100% cleaned up and generally runnable, but this is the code that we used for analyses figure generation for Detecting and harmonizing scanner differences in the ABCD study - annual release 1.0. We are running lots of permutations and the code here reflects writing those pemutations out to swarm files for use on the [NIH HPC](https://hpc.nih.gov/). If you're in another cluster computing environment, you may have to make change accordingly. 

#### Getting the data
In order to reproduce these analyses, you'll need to get the data from the [NIMH Data Archive](https://data-archive.nimh.nih.gov/). For this paper we are using [ABCD annual release 1.0](https://data-archive.nimh.nih.gov/abcd/query/annual-release-1.0.html), DOI 10.15154/1412097.  

#### Using these notebooks

These notebooks will take you through the analyses we ran. The order in which they should be run is:
1) prepare_abcd_data.ipynb
2) prep_run_swarm.ipynb
3) proc_perm_results_con.ipynb
4) proc_perm_results_task_based.ipynb

In between 2 and 3, you'll need to run the permutations, either on swarm, or some other way.

#### Rerunning the analysis with docker
To run the notebooks (using docker) clone this repository and cd into it on the command line then type the following with the appropriate absolute path to your data directory:

```
docker run -it -v [data_dir_absolute_path]:/data -v $PWD:/mnt --user root -e NB_GID=`id -u` -e NB_UID=`id -u` --rm -p 8888:8888 nihfmrif/abcd_combo start.sh jupyter lab
```

Now just paste the url listed in your terminal into your browser. You will only have access to the directory from which you ran the docker command and your data directory (located at /data).


#### Rerunning the analysis with singularity

Running the analysis using singularity is similar. To run a notebook server in a singularity container:

```
singularity exec -B [data_dir_absolute_path]:/data -B $PWD:/mnt -H ~/temp_for_singularity [path_to_singularity_image] start.sh jupyter lab --port=10104 --no-browser
```

One can create a singularity image in two ways. The first directly converts the docker image above into a singularity image. The advantage of this method is that it can be performed on the fly on a hpc environment. The second method must be performed on a machine with docker installed (this will not be possible within a high performance computing environment). The advantage of this latter method is that all data volumes of the host operating system can be mounted to the container.

1. On a host with singularity (no admin privileges) required:

```
singularity pull docker://nihfmrif/abcd_combo:bioarchive_submission_env
```

2. On a host with docker installed with the appropriate HPC directories listed after the "-m" flag so that they are added during the conversion process:

```
docker run -v /var/run/docker.sock:/var/run/docker.sock -v /data/rodgersleejg/general_singularity_images:/output --privileged -t --rm singularityware/docker2singularity -m "/gpfs /gs2 /gs3 /gs4 /gs5 /gs6 /gs7 /gs8 /gs9 /gs10 /gs11 /spin1 /data /scratch /fdb /lscratch"
```

To conveniently mount the directories in the step above one can use the SINGULARITY_BINDPATH variable. For example in bash on a slurm cluster:

```
export SINGULARITY_BINDPATH="/gs3,/gs4,/gs5,/gs6,/gs7,/gs8,/gs9,/gs10,/gs11,/spin1,/scratch,/fdb,/lscratch/$SLURM_JOB_ID:/tmp,/data"
```

### Deprecated:
Here's the commands to create a conda environment:

```
conda config --append channels bioconda
conda create -n abcd_mine python=3.6.3 jupyter notebook=5.4.1 r=3.3.1 rpy2=2.7.8 scikit-learn=0.19.1 patsy=0.4.1 pandas=0.20.3 numpy=1.13.3 statsmodels Bioconductor-sva Bioconductor-BiocParallel
. activate abcd_mine
pip install sklearn-pandas
```

