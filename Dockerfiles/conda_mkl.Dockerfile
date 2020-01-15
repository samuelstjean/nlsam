FROM continuumio/miniconda3:4.7.12

ENV DEPENDS_CONDA='cython==0.29 numpy==1.16.4 scipy==1.2.1 joblib==0.14.1' \
    DEPENDS_PIP='nibabel==2.4 dipy==0.15 autodmri==0.2.1 spams==2.6.1' \
    nlsam_version='0.6.1'

RUN apt update && \
    apt install libopenblas-dev g++ -y --no-install-recommends && \
    apt autoclean && \
    # get python deps
    conda install --yes --freeze-installed $DEPENDS_CONDA && \
    conda clean -afy && \
    pip install --no-cache-dir $DEPENDS_PIP && \
    # install nlsam itself
    # if you want to run the latest master instead use this link instead https://github.com/samuelstjean/nlsam/archive/master.zip
    pip install --no-cache-dir https://github.com/samuelstjean/nlsam/releases/download/v${nlsam_version}/nlsam-${nlsam_version}.tar.gz

# default command that will be run
CMD ["nlsam_denoising","--help"]
