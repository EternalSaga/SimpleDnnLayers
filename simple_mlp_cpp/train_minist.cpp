#include "load_mnist.h"
int main() {
  auto [trainSet,testSet] = RLDNN::loadMnist("D:\\ProgramAndStudy\\cpp_projects\\mnist-master");

  return 0;
}