#include "RLEigenUtils.h"
#include <exception>
#include <functional>
namespace RLDNN {
RandomChoice::RandomChoice() : rd(), randGen(rd()), idcesSet() {}

RLMask RandomChoice::operator()(size_t maxSize, size_t choiceNum) {
  std::uniform_int_distribution<> intDist(0, maxSize - 1);
  do {
    idcesSet.insert(intDist(randGen));
  } while (idcesSet.size() != choiceNum);
  RLMask mask(maxSize, 1);
  mask.setConstant(0);
  for (size_t i : idcesSet) {
    mask(i, 0) = 1;
  }
  this->idcesSet.clear();
  return mask;
}
MatrixXfRow reduceByMask(const MatrixXfRow& original, const RLMask& mask) {
  if (mask.cols() != 1) {
    throw std::invalid_argument("mask ought to be a col vector!");
  }
  size_t reduceSize((mask.array() > 0).count());
  MatrixXfRow returnV = MatrixXfRow::Zero(reduceSize, original.cols());
  size_t j(0);
  for (size_t i = 0; i < static_cast<size_t>(original.rows()); i++) {
    if (mask(i) > 0) {
      returnV.row(j) = original.row(i);
      j++;
    }
  }
  if (returnV.rows() != reduceSize) {
    throw std::runtime_error("input matrix has zero cols");
  }
  return returnV;
}
}  // namespace RLDNN