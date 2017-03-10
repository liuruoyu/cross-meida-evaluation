#!/usr/bin/env sh
set -e

TOOLS=/home/ruoyu/caffe/build/tools
MODEL=/home/ruoyu/Crossmedia\(deep\)/Models
PROTO=/home/ruoyu/Crossmedia\(deep\)/Networks
FEATURE=/home/ruoyu/Crossmedia\(deep\)/Features

$TOOLS/extract_features_binary.bin $MODEL/pascal_sentence/txt_network_1_iter_60.caffemodel $PROTO/pascal_sentence/ex/txt_deploy.prototxt fc3 $FEATURE/pascal_sentence/txt_feat1.nn 1000 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/pascal_sentence/txt_network_2_iter_60.caffemodel $PROTO/pascal_sentence/ex/txt_deploy.prototxt fc3 $FEATURE/pascal_sentence/txt_feat2.nn 1000 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/pascal_sentence/txt_network_3_iter_60.caffemodel $PROTO/pascal_sentence/ex/txt_deploy.prototxt fc3 $FEATURE/pascal_sentence/txt_feat3.nn 1000 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/pascal_sentence/txt_network_4_iter_60.caffemodel $PROTO/pascal_sentence/ex/txt_deploy.prototxt fc3 $FEATURE/pascal_sentence/txt_feat4.nn 1000 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/pascal_sentence/txt_network_5_iter_60.caffemodel $PROTO/pascal_sentence/ex/txt_deploy.prototxt fc3 $FEATURE/pascal_sentence/txt_feat5.nn 1000 GPU 0