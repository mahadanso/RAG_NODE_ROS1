FROM osrf/ros:noetic-desktop-full
ARG USERNAME=USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Delete user if it exists in container (e.g Ubuntu Noble: ubuntu)
RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# RUN apt-get update && apt-get upgrade -y
# Install software-properties-common to add the deadsnakes PPA
RUN apt-get update && apt-get install -y \
    software-properties-common

# Add the deadsnakes PPA, which provides newer versions of Python
RUN add-apt-repository ppa:deadsnakes/ppa

# Update and install Python 3.9 and its development headers
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-venv

# Create a virtual environment using the new Python version
RUN python3.9 -m venv /opt/venv

# Activate the new virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# RUN apt-get install -y python3-pip
    # libsqlite3-dev \
    # curl \
    # build-essential \
    # && rm -rf /var/lib/apt/lists/*

ENV SHELL /bin/bash

# Download, compile, and install a newer version of sqlite3.
# The version 3.47.0 is a recent example.
# RUN curl -L https://www.sqlite.org/2024/sqlite-autoconf-3470000.tar.gz | \
#     tar xz -C /tmp && \
#     cd /tmp/sqlite-autoconf-3470000 && \
#     ./configure && \
#     make && \
#     make install

# Update the linker cache so the new library is found.
# RUN ldconfig

# Set the LD_LIBRARY_PATH to ensure Python uses the new shared library.
# ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

# Initialize and update rosdep
RUN rosdep init || true && rosdep update

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
CMD ["/bin/bash"]