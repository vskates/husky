#!/bin/bash
set -e

# install/setup.bash может отсутствовать в свежем контейнере до первой сборки
# Подгружаем ROS 2
source /opt/ros/humble/setup.bash

# Подгружаем workspace
if [ -f /root/ros_ws/install/setup.bash ]; then
    source /root/ros_ws/install/setup.bash
fi

# Выполняем команду, которую передали
exec "$@"
