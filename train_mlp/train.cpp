#include "TrainMlp.h"

int main() {
  RLDNN::TrainMlp mlp("D:/ProgramAndStudy/cpp_projects/simple_mlp_cpp/simple_mlp_cpp/mlpConfig.json");
  mlp.startTrain();
  return 0;
}