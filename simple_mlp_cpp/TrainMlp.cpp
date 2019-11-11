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
  this->trainSize = trainSet.first.rows();
}

TrainMlp::~TrainMlp() {}

void TrainMlp::startTrain() {
  std::vector<float> trainLossList{};
  std::vector<float> trainAccList{};
  std::vector<float> testAccList{};
  auto iterPerEpoch = std::max(static_cast<int>(trainSize / batchSize), 1);
  RandomChoice randomChoice;
  auto network{MlpNet(MNIST_LENGTH, 50, LABEL_LENGTH)};
  for (size_t i = 0; i < iterationTimes; i++) {
    auto batchMask(randomChoice(trainSize, batchSize));

    MatrixXfRow xBatch;
    MatrixXfRow tBatch;
    try {
      xBatch = reduceByMask(trainSet.first, batchMask);
      tBatch = reduceByMask(trainSet.second, batchMask);
    } catch (const std::runtime_error& e) {
      std::cerr << e.what() << std::endl;
      std::abort();
    }

    auto grad = network.gradient(xBatch, tBatch);
    for (auto val : network.params) {
      network.params[val.first].array() -= grad[val.first].array() * this->learningRate;
    }
    auto loss{network.loss(xBatch, tBatch)};
    trainLossList.push_back(loss);
    std::cout << loss << std::endl;
  }
}
}  // namespace RLDNN

