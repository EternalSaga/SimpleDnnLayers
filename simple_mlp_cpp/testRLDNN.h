#pragma once
#include <Eigen/Core>
#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include "MlpNet.h"
#include "RLEigenUtils.h"
#include "SupportFunctions.h"
#include "load_mnist.h"
namespace RLDNN {
namespace TEST {
void testPredict();

}
}  // namespace RLDNN