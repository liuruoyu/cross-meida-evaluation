#!/usr/bin/env sh
set -e

TOOLS=/home/ruoyu/caffe/build/tools
MODEL=/home/ruoyu/Crossmedia\(deep\)/Models
PROTO=/home/ruoyu/Crossmedia\(deep\)/Networks
FEATURE=/home/ruoyu/Crossmedia\(deep\)/Features

$TOOLS/extract_features_binary.bin $MODEL/wiki/img_network_1_iter_240.caffemodel $PROTO/wiki/ex/img_deploy.prototxt fc8 $FEATURE/wiki/img_feat1.nn 2866 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/wiki/img_network_2_iter_240.caffemodel $PROTO/wiki/ex/img_deploy.prototxt fc8 $FEATURE/wiki/img_feat2.nn 2866 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/wiki/img_network_3_iter_180.caffemodel $PROTO/wiki/ex/img_deploy.prototxt fc8 $FEATURE/wiki/img_feat3.nn 2866 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/wiki/img_network_4_iter_240.caffemodel $PROTO/wiki/ex/img_deploy.prototxt fc8 $FEATURE/wiki/img_feat4.nn 2866 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/wiki/img_network_5_iter_240.caffemodel $PROTO/wiki/ex/img_deploy.prototxt fc8 $FEATURE/wiki/img_feat5.nn 2866 GPU 0