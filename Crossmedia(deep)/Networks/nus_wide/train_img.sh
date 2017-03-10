#!/usr/bin/env sh
set -e

LOG=/home/ruoyu/Crossmedia\(deep\)/Logs/nus_wide/train-`date +%Y-%m-%d-%H-%M-%S`.log
TOOLS=/home/ruoyu/caffe/build/tools
MODEL=/home/ruoyu/Crossmedia\(deep\)

$TOOLS/caffe train --solver=/home/ruoyu/Crossmedia\(deep\)/Networks/nus_wide/01/img_solver.prototxt --weights $MODEL/bvlc_reference_caffenet.caffemodel --gpu=1 2>&1 | tee $LOG
$TOOLS/caffe train --solver=/home/ruoyu/Crossmedia\(deep\)/Networks/nus_wide/02/img_solver.prototxt --weights $MODEL/bvlc_reference_caffenet.caffemodel --gpu=1 2>&1 | tee $LOG
$TOOLS/caffe train --solver=/home/ruoyu/Crossmedia\(deep\)/Networks/nus_wide/03/img_solver.prototxt --weights $MODEL/bvlc_reference_caffenet.caffemodel --gpu=1 2>&1 | tee $LOG
$TOOLS/caffe train --solver=/home/ruoyu/Crossmedia\(deep\)/Networks/nus_wide/04/img_solver.prototxt --weights $MODEL/bvlc_reference_caffenet.caffemodel --gpu=1 2>&1 | tee $LOG
$TOOLS/caffe train --solver=/home/ruoyu/Crossmedia\(deep\)/Networks/nus_wide/05/img_solver.prototxt --weights $MODEL/bvlc_reference_caffenet.caffemodel --gpu=1 2>&1 | tee $LOG
