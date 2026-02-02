FROM osrf/ros:humble-desktop-full

# -----------------------------------------
# Install system dependencies and tools
# -----------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-rosdep \
    git \
    cmake \
    build-essential \
    tree \
    vim \
    wget \
    socat \
    # CV system dependencies (ВАЖНО!)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # ROS2 image tools (КРИТИЧНО для CV!)
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    ros-humble-compressed-image-transport \
    ros-humble-vision-opencv \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------
# Install Python packages for CV/ML
# -----------------------------------------
RUN pip3 install --no-cache-dir \
    # PyTorch (обновленные версии)
    torch==2.1.0 \
    torchvision==0.16.0 \
    # OpenCV (обновленная версия + contrib)
    opencv-python==4.8.1.78 \
    opencv-contrib-python==4.8.1.78 \
    # Scientific computing
    numpy \
    scipy \
    # Utilities
    tqdm \
    pyyaml \
    pillow

# Initialize rosdep
RUN rosdep init || echo "rosdep already initialized" && rosdep update

# -----------------------------------------
# Create workspace
# -----------------------------------------
WORKDIR /root/ros_ws
RUN mkdir -p src

# Copy source code
COPY src/ ./src/

# Install ROS dependencies
RUN apt-get update && \
    /bin/bash -c "source /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y" && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------
# Build workspace
# -----------------------------------------
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release"

# -----------------------------------------
# Setup auto-sourcing for interactive shells
# -----------------------------------------
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "if [ -f /root/ros_ws/install/setup.bash ]; then source /root/ros_ws/install/setup.bash; fi" >> ~/.bashrc && \
    echo "echo ' ROS 2 Humble environment loaded " >> ~/.bashrc

# -----------------------------------------
# Entry point
# -----------------------------------------
COPY entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]