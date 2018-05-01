FROM jupyter/datascience-notebook:9f9e5ca8fe5a

USER root
RUN mkdir /data

USER $NB_UID

RUN conda install --quiet --yes \
    -c bioconda \
    Bioconductor-sva \
    Bioconductor-BiocParallel \
    scikit-learn=0.19.1 \
    pandas=0.20.3 \
    && conda clean -tipsy \

    && fix-permissions $CONDA_DIR \
    && fix-permissions /home/$NB_USER

RUN pip install --no-cache-dir sklearn-pandas

WORKDIR /mnt
