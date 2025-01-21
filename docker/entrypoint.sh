#!/bin/bash


export DISPLAY=:20
Xvfb :20 -screen 0 2000x1400x16 &
x11vnc -passwd survival -display :20 -N -forever &
fluxbox &
xinit &

cd /code
bash

