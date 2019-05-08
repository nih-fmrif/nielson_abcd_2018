FROM jupyter/datascience-notebook:2343e33dec46

USER root
RUN mkdir /data

USER $NB_UID

RUN conda uninstall --yes --quiet blas \
    && conda install --yes --quiet \
    --override-channels -c defaults \
    mkl=2019.3 
RUN conda install --yes --quiet \
    "blas=*=mkl"
RUN conda install --yes --quiet \
    -c bioconda \
    Bioconductor-sva=3.30.0 \
    Bioconductor-BiocParallel=1.16.6 \
    && conda install --yes --quiet \ 
    --override-channels -c defaults \
    scikit-learn=0.20.1 \
    pandas=0.23.4 \
    numpy=1.15.4 \
    statsmodels=0.9.0 \
    rpy2 \
    psutil \
    && conda clean -tipsy \
    && fix-permissions $CONDA_DIR \
    && fix-permissions /home/$NB_USER

RUN pip install --no-cache-dir sklearn-pandas

WORKDIR /mnt
