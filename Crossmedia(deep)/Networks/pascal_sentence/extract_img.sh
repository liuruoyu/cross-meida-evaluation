#!/usr/bin/env sh
set -e

TOOLS=/home/ruoyu/caffe/build/tools
MODEL=/home/ruoyu/Crossmedia\(deep\)/Models
PROTO=/home/ruoyu/Crossmedia\(deep\)/Networks
FEATURE=/home/ruoyu/Crossmedia\(deep\)/Features

$TOOLS/extract_features_binary.bin $MODEL/pascal_sentence/img_network_1_iter_60.caffemodel $PROTO/pascal_sentence/ex/img_deploy.prototxt fc8 $FEATURE/pascal_sentence/img_feat1.nn 1000 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/pascal_sentence/img_network_2_iter_60.caffemodel $PROTO/pascal_sentence/ex/img_deploy.prototxt fc8 $FEATURE/pascal_sentence/img_feat2.nn 1000 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/pascal_sentence/img_network_3_iter_60.caffemodel $PROTO/pascal_sentence/ex/img_deploy.prototxt fc8 $FEATURE/pascal_sentence/img_feat3.nn 1000 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/pascal_sentence/img_network_4_iter_60.caffemodel $PROTO/pascal_sentence/ex/img_deploy.prototxt fc8 $FEATURE/pascal_sentence/img_feat4.nn 1000 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/pascal_sentence/img_network_5_iter_60.caffemodel $PROTO/pascal_sentence/ex/img_deploy.prototxt fc8 $FEATURE/pascal_sentence/img_feat5.nn 1000 GPU 0