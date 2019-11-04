#include "TrainMlp.h"

namespace RLDNN {
TrainMlp::TrainMlp(std::string jsonConfigPath) {
  pt::ptree root;
  // Load the json file in this ptree
  pt::read_json(jsonConfigPath, root);
  this->mnistRootPath = root.get<std::string>("mnist root path");
  auto nextNode = root.get_child("train policy");
  this->batchSize = nextNode.get<size_t>("batch size");
  this->iterationTimes = nextNode.get<size_t>("iteration times");
  this->learningRate = nextNode.get<float>("learning rate");
  std::tie(trainSet, testSet) = loadMnist(mnistRootPath);
}

TrainMlp::~TrainMlp() {}

}  // namespace RLDNN
