#pragma once
#include <Eigen/Core>
#include <boost/endian/detail/endian_reverse.hpp>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
namespace fs = std::filesystem;
namespace RLDNN {
constexpr size_t MNIST_LENGTH{28 * 28};
constexpr size_t LABEL_LENGTH{10};
using TrainData =
    Eigen::Matrix<float, Eigen::Dynamic, MNIST_LENGTH, Eigen::RowMajor>;
using LabelData =
    Eigen::Matrix<float, Eigen::Dynamic, LABEL_LENGTH, Eigen::RowMajor>;
using MNistData = std::pair<TrainData, LabelData>;
std::tuple<MNistData, MNistData> loadMnist(fs::path mnistPath,
                                           bool normalize = true,
                                           bool flatten = true);
}  // namespace RLDNN
