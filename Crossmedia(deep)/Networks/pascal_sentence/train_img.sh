#!/usr/bin/env sh
set -e

LOG=/home/ruoyu/Crossmedia\(deep\)/Logs/pascal_sentence/train-`date +%Y-%m-%d-%H-%M-%S`.log
TOOLS=/home/ruoyu/caffe/build/tools
MODEL=/home/ruoyu/Crossmedia\(deep\)

$TOOLS/caffe train --solver=/home/ruoyu/Crossmedia\(deep\)/Networks/pascal_sentence/01/img_solver.prototxt --weights $MODEL/bvlc_reference_caffenet.caffemodel --gpu=1 2>&1 | tee $LOG
$TOOLS/caffe train --solver=/home/ruoyu/Crossmedia\(deep\)/Networks/pascal_sentence/02/img_solver.prototxt --weights $MODEL/bvlc_reference_caffenet.caffemodel --gpu=1 2>&1 | tee $LOG
$TOOLS/caffe train --solver=/home/ruoyu/Crossmedia\(deep\)/Networks/pascal_sentence/03/img_solver.prototxt --weights $MODEL/bvlc_reference_caffenet.caffemodel --gpu=1 2>&1 | tee $LOG
$TOOLS/caffe train --solver=/home/ruoyu/Crossmedia\(deep\)/Networks/pascal_sentence/04/img_solver.prototxt --weights $MODEL/bvlc_reference_caffenet.caffemodel --gpu=1 2>&1 | tee $LOG
$TOOLS/caffe train --solver=/home/ruoyu/Crossmedia\(deep\)/Networks/pascal_sentence/05/img_solver.prototxt --weights $MODEL/bvlc_reference_caffenet.caffemodel --gpu=1 2>&1 | tee $LOG
