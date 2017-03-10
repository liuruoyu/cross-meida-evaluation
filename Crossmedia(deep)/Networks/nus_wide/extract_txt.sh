#!/usr/bin/env sh
set -e

TOOLS=/home/ruoyu/caffe/build/tools
MODEL=/home/ruoyu/Crossmedia\(deep\)/Models
PROTO=/home/ruoyu/Crossmedia\(deep\)/Networks
FEATURE=/home/ruoyu/Crossmedia\(deep\)/Features

$TOOLS/extract_features_binary.bin $MODEL/nus_wide/txt_network_1_iter_5940.caffemodel $PROTO/nus_wide/ex/txt_deploy.prototxt fc3 $FEATURE/nus_wide/txt_feat1.nn 67994 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/nus_wide/txt_network_2_iter_5460.caffemodel $PROTO/nus_wide/ex/txt_deploy.prototxt fc3 $FEATURE/nus_wide/txt_feat2.nn 67994 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/nus_wide/txt_network_3_iter_5580.caffemodel $PROTO/nus_wide/ex/txt_deploy.prototxt fc3 $FEATURE/nus_wide/txt_feat3.nn 67994 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/nus_wide/txt_network_4_iter_5760.caffemodel $PROTO/nus_wide/ex/txt_deploy.prototxt fc3 $FEATURE/nus_wide/txt_feat4.nn 67994 GPU 0
$TOOLS/extract_features_binary.bin $MODEL/nus_wide/txt_network_5_iter_6420.caffemodel $PROTO/nus_wide/ex/txt_deploy.prototxt fc3 $FEATURE/nus_wide/txt_feat5.nn 67994 GPU 0