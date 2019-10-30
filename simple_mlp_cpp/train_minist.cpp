#include "load_mnist.h"
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
int main() {
  auto [trainSet,testSet] = RLDNN::loadMnist("D:\\ProgramAndStudy\\cpp_projects\\mnist-master");
  for (size_t i = 0; i < trainSet.size(); i++) {
    std::cout << trainSet[i].second << std::endl;
    cv::Mat numImg;
    cv::eigen2cv(trainSet[i].first, numImg);
    std::cout << trainSet[i].first << std::endl;
    cv::imshow("mnist", numImg);
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
  return 0;
}