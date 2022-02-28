FROM nvcr.io/nvidia/pytorch:21.05-py3
ENV DEBIAN_FRONTEND=noninteractive
# dependencies for gym
#
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 wget \
 vulkan-utils \
 mesa-vulkan-drivers \
 doxygen \
 graphviz \
 fonts-roboto \
 python3-sphinx \
 pigz \
 git \
 libegl1 \
 git-lfs \
 python3-pip \
 build-essential -y \
 python-dev \
 python-numpy \
 python3-dev \
 liboctomap-dev \
 libfcl-dev \
 freeglut3-dev \
 vim \
 libompl-dev \
 libeigen3-dev \
 liburdf-dev \
 libkdl-parser-dev \
 liborocos-kdl-dev \
 liburdfdom-dev \
 libnlopt-dev \
 swig \
 libnlopt-cxx-dev

RUN apt update \
&& wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb \
&& dpkg -i ./mesa_18.3.3-0.deb || true \
&& apt install -y -f

RUN pip install -q -U \
numpy-quaternion click tqdm sphinx sphinx-rtd-theme breathe pylama seaborn anytree trimesh pyrender

RUN pip3 install -q -U python-fcl
RUN pip3 install -q -U pybind11
RUN pip3 install -q -U pygame
RUN pip3 install -q -U wandb
RUN pip3 install -q -U gym
RUN pip3 install -U 'mujoco-py<2.2,>=2.1'
RUN pip3 install -q -U stable_baselines3
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

# COPY ./tracikpy /home/tracikpy
WORKDIR /home

ADD https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz /root/.mujoco/
RUN tar xvzf /root/user/.mujoco/mujoco210-linux-x86_64.tar.gz -C /root/.mujoco/
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

RUN git clone https://github.com/openai/mujoco-py.git
RUN cd /home/mujoco-py && pip install -e . --verbose

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
