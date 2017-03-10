#!/usr/bin/env sh
set -e

TOOLS=/home/ruoyu/caffe/build/tools
MODEL=/home/ruoyu/Crossmedia\(deep\)/Models
PROTO=/home/ruoyu/Crossmedia\(deep\)/Networks
FEATURE=/home/ruoyu/Crossmedia\(deep\)/Features

$TOOLS/extract_features_binary.bin $MODEL/nus_wide/img_network_1_iter_5940.caffemodel $PROTO/nus_wide/ex/img_deploy.prototxt fc8 $FEATURE/nus_wide/img_feat1.nn 67994 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/nus_wide/img_network_2_iter_5460.caffemodel $PROTO/nus_wide/ex/img_deploy.prototxt fc8 $FEATURE/nus_wide/img_feat2.nn 67994 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/nus_wide/img_network_3_iter_5580.caffemodel $PROTO/nus_wide/ex/img_deploy.prototxt fc8 $FEATURE/nus_wide/img_feat3.nn 67994 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/nus_wide/img_network_4_iter_5760.caffemodel $PROTO/nus_wide/ex/img_deploy.prototxt fc8 $FEATURE/nus_wide/img_feat4.nn 67994 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/nus_wide/img_network_5_iter_6420.caffemodel $PROTO/nus_wide/ex/img_deploy.prototxt fc8 $FEATURE/nus_wide/img_feat5.nn 67994 GPU 0