#pragma once
#include <Eigen/Core>
#include <Eigen/LU>
#include <random>
#include <set>
namespace RLDNN {
using namespace Eigen;
using MatrixXfRow = Matrix<float, Eigen::Dynamic, Eigen::Dynamic, RowMajor>;
template <typename T>
using MatrixXRow =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RLMask = MatrixXRow<int>;

class RandomChoice {
  std::random_device rd;
  std::set<size_t> idcesSet;
  std::mt19937 randGen;

 public:
  RandomChoice();
  std::set<size_t> operator()(size_t maxSize, size_t choiceNum);
};

enum Axis { COLUMN, ROW };

template <typename T>
void copySelfAlongSpecificAxis(MatrixXRow<T>& mat,
                               size_t copyTimes,
                               Axis axis) {
  // assert(copyTimes >= 2);
  // constexpr bool isFloatBool = std::is_same<
  // MatrixXRow<bool>,decltype(mat)>::value || std::is_same<
  // MatrixXRow<float>,decltype(mat)>::value; static_assert(isFloatBool);
  MatrixXRow<T> matCopy = mat;
  if (axis == COLUMN) {
    // A,B,C,D =>
    //|A,B,C,D|
    //|A,B,C,D|
    mat.conservativeResize(matCopy.rows() * copyTimes, Eigen::NoChange);
    for (size_t i = 1; i < copyTimes; i++) {
      // potential bugs
      for (size_t j = 0; j < static_cast<size_t>(matCopy.rows()); j++) {
        mat.row(i * matCopy.rows() + j) = matCopy.row(j);
      }
    }
  } else if (axis == ROW) {
    // A,B,C,D =>
    //|A,B,C,D| |A,B,C,D|
    mat.conservativeResize(Eigen::NoChange, matCopy.cols() * copyTimes);
    for (size_t i = 0; i < copyTimes; i++) {
      for (size_t j = 0; j < static_cast<size_t>(matCopy.cols()); j++) {
        mat.col(i * matCopy.cols() + j) = matCopy.col(j);
      }
    }
  } else {
    throw std::invalid_argument("axis must be ROW or COLUMN");
  }
}

MatrixXfRow reduceByRandChoice(const MatrixXfRow& original,
                         const std::set<size_t>& randSet);


std::tuple<VectorXi, VectorXi> getMaxIndexesValuesAccordingToRows(
    const MatrixXfRow& mat);
}  // namespace RLDNN