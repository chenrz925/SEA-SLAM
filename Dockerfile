FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Use Tsinghua source for apt-get
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# Install ROS Noetic
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y curl gnupg2 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://mirror.ghproxy.com/https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc -o /tmp/ros.asc && apt-key add /tmp/ros.asc && rm /tmp/ros.asc
RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list
RUN apt-get update && apt-get install -y tzdata
RUN apt-get install -y \
    ros-noetic-desktop-full \
    && rm -rf /var/lib/apt/lists/*

# Set up ROS environment
ENV ROS_DISTRO noetic
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc

# Setup ORB-SLAM3 dependencies
# Install OpenCV 4.5.2
RUN cd /tmp && apt-get update && apt-get install -y cmake g++ wget unzip make \
    && wget -O opencv.zip https://mirror.ghproxy.com/https://github.com/opencv/opencv/archive/4.5.2.zip \
    && wget -O opencv_contrib.zip https://mirror.ghproxy.com/https://github.com/opencv/opencv_contrib/archive/4.5.2.zip \
    && unzip opencv.zip \
    && unzip opencv_contrib.zip \
    && mkdir -p build && cd build \
    && cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.5.2/modules ../opencv-4.5.2 \
    && cmake --build . \
    && make -j4 \
    && make install \
    && rm -rf /tmp/*
# Install Pangolin 0.6
RUN cd /tmp && curl https://mirror.ghproxy.com/https://github.com/stevenlovegrove/Pangolin/archive/v0.6.zip --output Pangolin.zip \
    && unzip Pangolin.zip && mkdir Pangolin-0.6/build && cd Pangolin-0.6/build \
    && cmake .. \
    && cmake --build . \
    && make -j4 && make install \
    && rm -rf /tmp/*
# Install Eigen 3.4.0
RUN cd /tmp && curl https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip --output eigen-3.4.0.zip \
    && unzip eigen-3.4.0.zip && mkdir eigen-3.4.0/build && cd eigen-3.4.0/build \
    && cmake .. \
    && make install \
    && rm -rf /tmp/*

RUN apt-get install -y libgoogle-glog-dev

# Install Python dependencies
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 150 \
    && apt-get install -y python3-pip \
    && pip install cython \ 
    && pip install pillow pycocotools matplotlib 'networkx~=3.1' ncnn \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip uninstall -y opencv-python

# Add source code
COPY . /root/catkin_ws
WORKDIR /root/catkin_ws
RUN chmod a+x /root/catkin_ws/src/SemanticCNN/script/server.py \
    && /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make" \
    && echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

# ROS entrypoint
CMD ["/bin/bash", "-c", "source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash && roscore"]