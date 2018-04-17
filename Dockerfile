FROM jupyter/datascience-notebook:a2875abd3fc3

USER root
RUN mkdir /data

USER $NB_UID

RUN conda install --quiet --yes \
    -c bioconda \
    Bioconductor-sva \
    Bioconductor-BiocParallel && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

RUN pip install --no-cache-dir sklearn-pandas

WORKDIR /mnt
