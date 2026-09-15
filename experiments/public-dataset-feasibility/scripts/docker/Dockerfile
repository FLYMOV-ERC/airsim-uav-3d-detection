# ROS 1 Noetic on Ubuntu 20.04, built to read the Holybro rosbags on macOS.
#
# This container is the environment record for extract_frames_from_rosbag.py,
# detect_in_rosbag.py and plot_mocap_trajectory.py -- all three need rosbag and
# cv_bridge, which have no usable macOS build.
#
# Note the CPU-only torch index below: inference inside the container runs on the
# CPU. That is why training (train_yolov8.py) was run natively on the host
# against Apple MPS instead, which is what the `device: mps` field in
# results/run-args/*.yaml records.
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Basic packages
RUN apt update && apt install -y curl gnupg2 lsb-release wget

# ROS repository key
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# ROS repository
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# ROS and the message/bridge packages the scripts import
RUN apt update && apt install -y \
    ros-noetic-ros-base \
    ros-noetic-cv-bridge \
    ros-noetic-rosbag \
    ros-noetic-sensor-msgs \
    ros-noetic-geometry-msgs \
    python3-pip \
    python3-dev \
    python3-opencv \
    wget \
    git && \
    apt clean

# Python packages. torch is installed from the CPU-only wheel index.
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install ultralytics opencv-python numpy

# Source ROS on every shell
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# Working directory. docker-compose.yml mounts the project root -- the
# directory holding README.md, scripts/ and results/ -- here, so the scripts
# are at /workspace/scripts/*.py inside the container.
WORKDIR /workspace

CMD ["/bin/bash"]
