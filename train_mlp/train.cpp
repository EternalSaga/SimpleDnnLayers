#include "TrainMlp.h"

int main() {
  RLDNN::TrainMlp mlp("D:\\ProgramAndStudy\\cpp_projects\\simple-two-layers-mlp\\simple_mlp_cpp\\mlpConfig.json");
  mlp.startTrain();
  return 0;
}