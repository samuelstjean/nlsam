FROM python:3.7-buster

ENV DEPENDS='cython==0.29 nibabel==2.4 dipy==0.15 numpy==1.16.4 scipy==1.2.2 joblib==0.14.1 autodmri==0.2.3' \
    DEPENDS_spams='https://github.com/samuelstjean/spams-python/releases/download/v2.6.1/spams-2.6.tar.gz' \
    nlsam_version='0.6.1'

RUN apt update && \
    apt install libopenblas-dev -y --no-install-recommends && \
    apt autoclean && \
    # get python deps
    pip3 install --no-cache-dir $DEPENDS $DEPENDS_spams && \
    # install nlsam itself
    # if you want to run the latest master instead use this link instead https://github.com/samuelstjean/nlsam/archive/master.zip
   pip3 install --no-cache-dir https://github.com/samuelstjean/nlsam/releases/download/v${nlsam_version}/nlsam-${nlsam_version}.tar.gz

# default command that will be run
CMD ["nlsam_denoising","--help"]
