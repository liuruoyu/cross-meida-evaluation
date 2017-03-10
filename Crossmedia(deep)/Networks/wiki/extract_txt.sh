#!/usr/bin/env sh
set -e

TOOLS=/home/ruoyu/caffe/build/tools
MODEL=/home/ruoyu/Crossmedia\(deep\)/Models
PROTO=/home/ruoyu/Crossmedia\(deep\)/Networks
FEATURE=/home/ruoyu/Crossmedia\(deep\)/Features

$TOOLS/extract_features_binary.bin $MODEL/wiki/txt_network_1_iter_240.caffemodel $PROTO/wiki/ex/txt_deploy.prototxt fc3 $FEATURE/wiki/txt_feat1.nn 2866 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/wiki/txt_network_2_iter_240.caffemodel $PROTO/wiki/ex/txt_deploy.prototxt fc3 $FEATURE/wiki/txt_feat2.nn 2866 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/wiki/txt_network_3_iter_180.caffemodel $PROTO/wiki/ex/txt_deploy.prototxt fc3 $FEATURE/wiki/txt_feat3.nn 2866 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/wiki/txt_network_4_iter_240.caffemodel $PROTO/wiki/ex/txt_deploy.prototxt fc3 $FEATURE/wiki/txt_feat4.nn 2866 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/wiki/txt_network_5_iter_240.caffemodel $PROTO/wiki/ex/txt_deploy.prototxt fc3 $FEATURE/wiki/txt_feat5.nn 2866 GPU 0