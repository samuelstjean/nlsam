FROM continuumio/miniconda3:4.7.12

ENV DEPENDS_CONDA='cython==0.29 numpy==1.16.4 scipy==1.2.1 joblib==0.14.1 mkl==2019.1 mkl-include==2019.1 intel-openmp==2019.1' \
    DEPENDS_PIP='nibabel==2.4 dipy==0.15 autodmri==0.2.1 distro==1.5.0' \
    nlsam_version='0.6.1' \
    DEPENDS_spams='https://github.com/samuelstjean/spams-python/releases/download/v2.6.1/spams-2.6.tar.gz'

RUN apt update && \
    apt install g++ -y --no-install-recommends && \
    apt autoclean && \
    # get python deps
    conda install --yes --freeze-installed $DEPENDS_CONDA && \
    conda clean -afy && \
    pip install --no-cache-dir $DEPENDS_PIP  && \
    # We disable pyproject.toml here to link to intel mkl explicitly, this is also why we installed distro before explicitly
    pip install --no-cache-dir $DEPENDS_spams --no-build-isolation && \
    # install nlsam itself
   if [ "$nlsam_version" = "master" ]; \
        then pip install --no-cache-dir https://github.com/samuelstjean/nlsam/archive/master.zip; \
        else pip3 install --no-cache-dir https://github.com/samuelstjean/nlsam/releases/download/v${nlsam_version}/nlsam-${nlsam_version}.tar.gz; \
    fi;

# default command that will be run
ENTRYPOINT ["nlsam_denoising"]
CMD ["--help"]
