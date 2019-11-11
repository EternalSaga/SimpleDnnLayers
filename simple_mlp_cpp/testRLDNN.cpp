#include <Eigen/Core>
#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include "SupportFunctions.h"
#include "load_mnist.h"
#include "RLEigenUtils.h"
namespace RLDNN {
namespace TEST {
	using std::cout;
	using std::endl;
void testRandomChoice() {
  RandomChoice randomChoice{};
  RLMask mask{randomChoice(200,30)};
  std::cout << mask << std::endl;
  assert((mask.array() > 0).count() == 30);
}

void testLoadMnist() {
  auto [trainSet, testSet] =
      RLDNN::loadMnist("D:\\ProgramAndStudy\\cpp_projects\\mnist-master");
  for (size_t i = 0; i < trainSet.first.rows(); i++) {
    float label{0};
    trainSet.second.row(i).maxCoeff(&label);
    std::cout << label << std::endl;
    cv::Mat numImg;

    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> xx(trainSet.first.row(i));

	xx.resize(28, 28);
    cv::eigen2cv(xx, numImg);
    cv::imshow("mnist", numImg);
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
}

void testSupportFunctions() {
  MatrixXfRow testData(4, 3);
  testData << 100, 200, 300, 550, 200, 250, 1.3, 4.5, 0.33, 332, 423, 1.1;
  std::cout << testData << std::endl;
  MatrixXfRow sigmoidExpected(4, 3);  // Row 4, Col 3
  sigmoidExpected << 1., 1., 1., 1., 1., 1., 0.78583498, 0.98901306, 0.58175938,
      1., 1., 0.75026011;
  MatrixXfRow softmaxExpected(4, 3);
  softmaxExpected << 0.0000000e+00, 3.7835059e-44, 1.0000000e+00,
 1.0000000e+00 ,0.0000000e+00,0.0000000e+00,
 3.8592733e-02, 9.4677740e-01 ,1.4629850e-02,
 3.0144032e-40,1.0000000e+00,0.0000000e+00;
  auto softMaxOut(softmax(testData));
  std::cout << softMaxOut << std::endl;
  assert(softmaxExpected.isApprox(softMaxOut));

  auto sigmoidOut(sigmoid(testData));
  std::cout << sigmoidOut << std::endl;
  assert(sigmoidExpected.isApprox(sigmoidOut));

  MatrixXfRow testTruth(2, 4);
  MatrixXfRow testPredict(2, 4);
  testTruth << 0,0,0,1,
        0,1,0,0;
  testPredict << 0.1,0.3,0.4,0.2,
            0.7,0.1,0.1,0.1;
  float expectedLoss{1.9560108184814453f};
  auto loss = crossEntropyError(testPredict, testTruth);
  assert(expectedLoss == loss);
}
void testReduceByMask() {
  RLMask mask(7, 1);
  mask << 1, 0, 0, 0, 1, 1, 0;
  auto toBeMask = MatrixXfRow(7, 3);
  toBeMask << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
      20, 21;

  auto result = reduceByMask(toBeMask, mask);
  auto expected = MatrixXfRow(3, 3);
  expected << 1, 2, 3, 13, 14, 15, 16, 17, 18;
  assert(expected.isApprox(result));
}
}  // namespace TEST
}  // namespace RLDNN


int main() {
  using namespace RLDNN::TEST;
  testReduceByMask();
  //testLoadMnist();
  return 0;
}