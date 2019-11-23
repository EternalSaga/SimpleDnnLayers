#include "TrainMlp.h"
#include <algorithm>
#include "TimeStamp.hpp"
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
  Eigen::initParallel();
  auto iterPerEpoch = std::max(static_cast<int>(trainSize / batchSize), 1);
  // auto iterPerEpoch = 10;
  RandomChoice randomChoice;
  auto network{MlpNet(MNIST_LENGTH, 50, LABEL_LENGTH)};
  float time{0.f};
  RLVulkan::TimeStamp ts{};
  
  for (size_t i = 0; i < iterationTimes; i++) {
    ts.setStart();
    std::set<size_t> randSet(randomChoice(trainSize, batchSize));
    std::vector<int> indecs(randSet.begin(), randSet.end());

    MatrixXfRow xBatch = trainSet.first(indecs, Eigen::all);
    MatrixXfRow tBatch = trainSet.second(indecs, Eigen::all);

    auto grad = network.gradient(xBatch, tBatch);
    for (auto val : network.params) {
      network.params[val.first].array() -=
          grad[val.first].array() * this->learningRate;
    }
    auto loss{network.loss(xBatch, tBatch)};
    ts.setEnd();
    time += ts.getElapsedTime<std::chrono::milliseconds>();
    if (i % iterPerEpoch == 0) {
      trainLossList.push_back(loss);
      std::cout << loss << std::endl;
      float trainAcc = network.getAccuracy(trainSet.first, trainSet.second);
      trainAccList.push_back(trainAcc);
      float testAcc = network.getAccuracy(testSet.first, testSet.second);
      testAccList.push_back(testAcc);
      std::cout << "Train Acc | " << trainAcc << ". Test Acc | " << testAcc
                << std::endl;
    }
  }
  std::cout << "Average time per iteration:" << time / this->iterationTimes
            << std::endl;
}

}  // namespace RLDNN
