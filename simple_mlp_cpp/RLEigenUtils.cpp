#include "RLEigenUtils.h"
#include <exception>
#include <functional>
namespace RLDNN {
RandomChoice::RandomChoice() : rd(), randGen(rd()), idcesSet() {}

std::set<size_t> RandomChoice::operator()(size_t maxSize, size_t choiceNum) {
  std::uniform_int_distribution<> intDist(0, maxSize - 1);
  do {
    idcesSet.insert(intDist(randGen));
  } while (idcesSet.size() != choiceNum);
  auto randSet = idcesSet;
  this->idcesSet.clear();
  return randSet;
}
MatrixXfRow reduceByRandChoice(const MatrixXfRow& original,
                               const std::set<size_t>& randSet) {
  std::vector<int> indecs(randSet.begin(), randSet.end());

 MatrixXfRow returnV = original(indecs, Eigen::all);

  return returnV;
}
std::tuple<VectorXi, VectorXi> getMaxIndexesValuesAccordingToRows(
    const MatrixXfRow& mat) {
  VectorXi maxIndexes(mat.rows());
  VectorXi maxValues(mat.rows());
  for (size_t i = 0; i < static_cast<size_t>(mat.rows()); i++) {
    maxValues(i) = mat.row(i).maxCoeff(&maxIndexes(i));
  }
  return std::make_tuple(maxIndexes, maxValues);
}
}  // namespace RLDNN