#pragma once
#include <Eigen/Core>
#include <boost/endian/detail/endian_reverse.hpp>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
namespace RLDNN {
namespace fs = std::filesystem;
std::tuple<std::vector<std::pair<Eigen::Matrix<float, 28, 28, Eigen::RowMajor>,
                                 Eigen::RowVectorXf>>,
           std::vector<std::pair<Eigen::Matrix<float, 28, 28, Eigen::RowMajor>,
                                 Eigen::RowVectorXf>>>
loadMnist(fs::path mnistPath, bool normalize = true, bool flatten = true);
}  // namespace RLDNN
