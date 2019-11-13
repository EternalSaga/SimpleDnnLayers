#pragma once
#include "MlpNet.h"
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
namespace RLDNN {
namespace pt = boost::property_tree;

class TrainMlp {

 public:
  TrainMlp(std::string jsonConfigPath);
  ~TrainMlp();
  void startTrain();
 private:
  MNistData trainSet;
  MNistData testSet;

  std::string mnistRootPath;
  size_t iterationTimes;
  size_t batchSize;
  float learningRate;
  size_t trainSize;

};
}  // namespace RLDNN
