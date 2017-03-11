# A New Evaluation Protocol and Benchmarking Results for Extendable Cross-media Retrieval
Ruoyu Liu, Yao Zhao, Liang Zheng, Shikui Wei, Yi Yang

## Introduction
Cross-media retrieval studies the problem that the modelities of the query and database are not the same. In previous works, the performance of cross-media is tested on the classical datasets and the train/test classes are identical. However, the query image/text query may exhibit various content and it is challenging for the training process to take into all the variety in query types. It is therefore not well-grounded for an evaluaiton protocol to assume that the train/test data have the same set of classes. We propose a new evaluation protocol of the extendable cross-media retrieval and re-evaluate the performance of the baseline methods.

Detailed description is provided in our paper.

## System Requirements
This software is both tested on Windows 8.1 and Ubuntu 16.04 LTS (64bit).

MATLAB (tested with 2015a on Windows and 2016b on 64-bit Linux).

[Caffe](http://caffe.berkeleyvision.org/installation.html#prequequisites) (also provided in our project).

## Demo
You can directly run the three script files in "crossmedia" to view the results of real-valed representation methods, binary representition methods and trivial solution methods. The results are also saved in this folder.
```
>> run runrl.m
>> run runhs.m
>> run runts.m
```

## Train the Deep Models
To train the deep models tested in our paper. You need to download the data of the three benchmark datasets from [here](http://pan.baidu.com/s/1pLBCTK3) (password: *ja9t*), unzip the file and put the inside folders into "Benchmarks". Then, you need to download the pre-trained CaffeNet and the image mean value file from [here](http://pan.baidu.com/s/1i5yrX5b) (password: *8krs*). Unzip the file and the put the inside two files into "Crossmedia(deep)".

Modify the data path variables of the script files in "Crossmedia(deep)/Codes", then run these files to generate the training data.

Modify the data path of the ".sh" and ".prototxt" files in "Crossmedia(deep)/Networks", and then run the ".sh" files to train the models and extract the features.

Copy the new feature files (".nn") to the folder "Crossmedia/deep_features".

## Result Files
You can download the result figures illustrated in our paper (saved as ".fig" files) from "Figs" for direct comparison.
