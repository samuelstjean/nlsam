FROM python:3.5-buster

ENV DEPENDS='cython==0.29 nibabel==2.4 dipy==0.15 numpy==1.16.4 scipy==1.2.2 joblib==0.14.1' \
    # this one is for py3.6, which is what ubuntu 18.04 is using. Feel free to change for a different version as appropriate.
    DEPENDS_spams='https://github.com/samuelstjean/spams-python/releases/download/v2.6/spams-2.6-cp35-cp35m-linux_x86_64.whl' \
    nlsam_version='0.6.1'

RUN apt update && \
    apt install libopenblas-base -y --no-install-recommends && \
    apt autoclean && \
    # get python deps
    pip3 install --no-cache-dir $DEPENDS $DEPENDS_spams && \
    # install nlsam itself
    # if you want to run the latest master instead use this link instead https://github.com/samuelstjean/nlsam/archive/master.zip
    pip3 install --no-cache-dir https://github.com/samuelstjean/nlsam/releases/download/v${nlsam_version}/nlsam-${nlsam_version}.tar.gz

# default command that will be run
CMD ["nlsam_denoising","--help"]
