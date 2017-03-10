#!/usr/bin/env sh
set -e

LOG=/home/ruoyu/Crossmedia\(deep\)/Logs/pascal_sentence/train-`date +%Y-%m-%d-%H-%M-%S`.log
TOOLS=/home/ruoyu/caffe/build/tools
MODEL=/home/ruoyu/Crossmedia\(deep\)

$TOOLS/caffe train --solver=/home/ruoyu/Crossmedia\(deep\)/Networks/pascal_sentence/01/txt_solver.prototxt --gpu=1 2>&1 | tee $LOG
$TOOLS/caffe train --solver=/home/ruoyu/Crossmedia\(deep\)/Networks/pascal_sentence/02/txt_solver.prototxt --gpu=1 2>&1 | tee $LOG
$TOOLS/caffe train --solver=/home/ruoyu/Crossmedia\(deep\)/Networks/pascal_sentence/03/txt_solver.prototxt --gpu=1 2>&1 | tee $LOG
$TOOLS/caffe train --solver=/home/ruoyu/Crossmedia\(deep\)/Networks/pascal_sentence/04/txt_solver.prototxt --gpu=1 2>&1 | tee $LOG
$TOOLS/caffe train --solver=/home/ruoyu/Crossmedia\(deep\)/Networks/pascal_sentence/05/txt_solver.prototxt --gpu=1 2>&1 | tee $LOG
