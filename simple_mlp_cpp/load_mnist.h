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
using MNistData =
    std::vector<std::pair<Eigen::Matrix<float, 28, 28, Eigen::RowMajor>,
                          Eigen::RowVectorXf>>;
std::tuple<MNistData, MNistData> loadMnist(fs::path mnistPath,
                                           bool normalize = true,
                                           bool flatten = true);
}  // namespace RLDNN
