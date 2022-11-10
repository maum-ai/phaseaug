From nvcr.io/nvidia/pytorch:20.11-py3

Expose 6006 6007 6008 6009

RUN python3 -m pip uninstall -y \
    tensorboard \
    tensorboard-plugin-dlprof \
    nvidia-tensorboard \
    nvidia-tensorboard-plugin-dlprof \
    jupyter-tensorboard \
    && \
python3 -m pip --no-cache-dir install --upgrade \
    tensorboard==2.0.0 \
    omegaconf==2.1.0 \
    gpustat==0.6.0 \
    grpcio==1.13.0 \
    grpcio-tools==1.13.0 \
    protobuf==3.6.0 \
    Cython==0.29.21 \
    librosa==0.8.0 \
    matplotlib==3.3.1 \
    numpy==1.18.5 \
    scipy==1.5.2 \
    Unidecode==1.1.1 \
    kiwipiepy==0.8.1 \
    alias_free_torch==0.0.6 \
    && \
apt update && \
apt install -y \
    tmux \
    htop \
    ncdu \
    vim \
    && \
apt clean && \
apt autoremove && \
rm -rf /var/lib/apt/lists/* /tmp/* && \
mkdir /hifigan
WORKDIR /hifigan

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID jun3518
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID jun3518

USER jun3518
