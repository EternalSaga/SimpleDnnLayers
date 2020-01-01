#pragma once
#include <Eigen/Core>
#include <boost/endian/detail/endian_reverse.hpp>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#ifdef __linux__
    #if __GNUC__ <8 && __GNUC__ >=7
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
    #elif __GNUC__ >=8
    #include <filesystem>
    namespace fs = std::filesystem;
    #else
    #error "Please update your gcc to gcc7 or upper version"
    #endif
#endif

#ifdef _WIN64
    #if _MSC_VER >= 1910
    #include <filesystem>
    namespace fs = std::filesystem;
    #else
    #error "Please update your vs to vs2017 or upper version"
    #endif
#endif

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
