FROM gitpod/workspace-full

RUN sudo apt-get update \
 && sudo apt-get install -y \
    tool \
    gfortran \
    libblas-dev liblapack-dev \
 && sudo rm -rf /var/lib/apt/lists/*
