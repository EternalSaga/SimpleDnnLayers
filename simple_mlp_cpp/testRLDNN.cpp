#include <Eigen/Core>
#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include "SupportFunctions.h"
#include "load_mnist.h"
namespace RLDNN {
namespace TEST {
void testLoadMnist() {
  auto [trainSet, testSet] =
      RLDNN::loadMnist("D:\\ProgramAndStudy\\cpp_projects\\mnist-master");
  for (size_t i = 0; i < trainSet.size(); i++) {
    std::cout << trainSet[i].second << std::endl;
    cv::Mat numImg;
    cv::eigen2cv(trainSet[i].first, numImg);
    std::cout << trainSet[i].first << std::endl;
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
}

}  // namespace TEST
}  // namespace RLDNN

int main() {
  using namespace RLDNN::TEST;
  testSupportFunctions();
  return 0;
}