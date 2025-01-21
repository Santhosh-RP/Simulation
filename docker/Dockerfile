# docker build -f docker/Dockerfile -t survival .
# docker run -it -v .:/code -p 127.0.0.1:5920:5920 -p 127.0.0.1:6006:6006 --rm --gpus all survival bash /code/docker/entrypoint.sh
# xtightvncviewer localhost:5920

#
# 3. See results:
# tensorboard --logdir ~/logdir
# System
FROM ubuntu:jammy
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

RUN apt-get update
RUN apt-get install -y ffmpeg git python3-pip vim libglew-dev x11-xserver-utils xvfb
RUN apt-get install -y cmake python3-pyside2* x11vnc xvfb fluxbox
RUN apt-get install -y mc
RUN apt-get clean


RUN pip3 install --upgrade pip
RUN pip3 install numpy==1.26.1
RUN pip3 install tensorboardx==2.6.1
RUN pip3 install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install pygame==2.5.2
RUN pip3 install PySide2==5.15.2.1
RUN pip3 install PyQt5
RUN pip3 install pyqtgraph==0.13.3
RUN pip3 install PyYAML==6.0.1
RUN pip3 install gymnasium==0.29.1 glfw==2.5.9 imageio==2.32.0
RUN pip3 install mujoco==2.3.5 matplotlib==3.8.1
RUN pip3 install cma==3.3.0
RUN pip3 install tensorboard==2.15.1
RUN pip3 install stable_baselines3==2.1.0


# RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2/qt

RUN ln -s /usr/bin/python3.10 /usr/bin/python

# VNC server
EXPOSE 5920
ENV DISPLAY :20
RUN echo "exec fluxbox" >> ~/.xinitrc
RUN chmod +x ~/.xinitrc
RUN mkdir ~/.fluxbox
RUN echo "[startup] {xterm -e 'cd /code && /bin/bash'}" >> ~/.fluxbox/apps
RUN echo "[startup] {xterm -e 'cd /code && /bin/bash'}" >> ~/.fluxbox/apps
RUN echo "[exec] (xterm) {xterm -e 'cd /code && /bin/bash'}" >> ~/.fluxbox/menu

RUN apt-get update
RUN apt-get install -y terminator firefox
RUN pip3 install wandb
RUN pip3 install opencv-python
#-headless==4.8.1.78 --force
