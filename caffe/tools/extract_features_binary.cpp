#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <boost/make_shared.hpp>
#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include <boost/lexical_cast.hpp>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
//#include "caffe/dataset_factory.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp" 

using namespace boost;
using namespace std;

using boost::shared_ptr;
using caffe::Blob;
using caffe::Caffe;
//using caffe::Dataset;
//using caffe::DatasetFactory;
using caffe::Datum;
using caffe::Net;
using std::ofstream;

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 6;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_features  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
    "  save_feature_dataset_name1[,name2,...]  num_mini_batches "
    "  [CPU/GPU] [DEVICE_ID=0]\n"
    "Note: you can extract multiple features in one pass by specifying"
    " multiple feature blob names and dataset names seperated by ','."
    " The names cannot contain white space characters and the number of blobs"
    " and datasets must be equal.";
    return 1;
  }
  int arg_pos = num_required_args;

  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    uint device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  arg_pos = 0;  // the name of the executable
  std::string pretrained_binary_proto(argv[++arg_pos]);

  // Expected prototxt contains at least one data layer such as
  //  the layer data_layer_name and one feature blob such as the
  //  fc7 top blob to extract features.
  /*
   layers {
     name: "data_layer_name"
     type: DATA
     data_param {
       source: "/path/to/your/images/to/extract/feature/images_leveldb"
       mean_file: "/path/to/your/image_mean.binaryproto"
       batch_size: 128
       crop_size: 227
       mirror: false
     }
     top: "data_blob_name"
     top: "label_blob_name"
   }
   layers {
     name: "drop7"
     type: DROPOUT
     dropout_param {
       dropout_ratio: 0.5
     }
     bottom: "fc7"
     top: "fc7"
   }
   */

  std::string feature_extraction_proto(argv[++arg_pos]);
  shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto, caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  string extract_feature_blob_names(argv[++arg_pos]);
    vector<string> blob_names;
    boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

    string save_feature_leveldb_names(argv[++arg_pos]);
    vector<string> leveldb_names;
    boost::split(leveldb_names, save_feature_leveldb_names,
                 boost::is_any_of(","));
    CHECK_EQ(blob_names.size(), leveldb_names.size()) <<
        " the number of blob names and leveldb names must be equal";
    size_t num_features = blob_names.size();

    for (size_t i = 0; i < num_features; i++) {
      CHECK(feature_extraction_net->has_blob(blob_names[i]))
          << "Unknown feature blob name " << blob_names[i]
          << " in the network " << feature_extraction_proto;
    }

    vector<FILE* > feature_dbs;
    for (size_t i = 0; i < num_features; ++i) {
        LOG(INFO)<< "Opening save file: " << leveldb_names[i];
		FILE* fp = fopen(leveldb_names[i].c_str(), "wb");
		CHECK(fp) << leveldb_names[i];
		feature_dbs.push_back(fp);
    }

    int num_mini_batches = atoi(argv[++arg_pos]);

  LOG(ERROR)<< "Extacting Features";
  caffe::Timer timer;
  timer.Start();

  std::vector<Blob<float>*> input_vec;
  std::vector<int> image_indices(num_features, 0);
  //每个batchsize一个循环
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {

	  feature_extraction_net->Forward(input_vec);
    //每个batchsize中，要提取num_features个layer的特征
    for (int i = 0; i < num_features; ++i) {
    	LOG(ERROR) << "Extracting the " << batch_index <<"th image feature";
		if (timer.MilliSeconds() >= 2000) {
			LOG(INFO) << batch_index;
			timer.Start();
		 }
         const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[i]);
         int batch_size = feature_blob->num();
         int dim_features = feature_blob->count() / batch_size;
         if (batch_index == 0)
		  {
			 int total_num = num_mini_batches * batch_size;
			 CHECK_EQ(fwrite(&total_num, sizeof(int), 1, feature_dbs[i]), 1);
			 CHECK_EQ(fwrite(&dim_features, sizeof(int), 1, feature_dbs[i]), 1);
		  }
         const Dtype* p_data = feature_blob->cpu_data();
         CHECK_EQ(fwrite(p_data, sizeof(Dtype), feature_blob->count(), feature_dbs[i]), feature_blob->count());

    }  // for (int i = 0; i < num_features; ++i)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  // write the last batch
  for (int i = 0; i < num_features; ++i) {
	  fclose(feature_dbs[i]);
  }

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}

