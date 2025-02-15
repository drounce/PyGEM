FROM ubuntu:24.04

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo curl vim git tree python3-pip python3-venv python3-dev build-essential \
    libhdf5-dev libnetcdf-dev libproj-dev gdal-bin libgdal-dev libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and add to sudo group
RUN usermod -aG sudo ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu

# Switch to non-root user
USER ubuntu
WORKDIR /home/ubuntu

# Create a virtual environment and install PyGEM in it
RUN python3 -m venv pygem_venv && \
    . pygem_venv/bin/activate && \
    git clone https://github.com/PyGEM-Community/PyGEM.git && \
    pip install -e PyGEM && \
    initialize

# Clone the PyGEM notebooks repository, which are used for testing
RUN git clone https://github.com/PyGEM-Community/PyGEM-notebooks.git