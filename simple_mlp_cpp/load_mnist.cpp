#include "load_mnist.h"

std::tuple<std::vector<std::pair<Eigen::Matrix<float, 28, 28, Eigen::RowMajor>,
                                 Eigen::RowVectorXf>>,
           std::vector<std::pair<Eigen::Matrix<float, 28, 28, Eigen::RowMajor>,
                                 Eigen::RowVectorXf>>>
RLDNN::loadMnist(fs::path mnistPath, bool normalize, bool flatten) {
  using namespace Eigen;
  using boost::endian::endian_reverse_inplace;
  const fs::path trainImagesPath{"train-images-idx3-ubyte"};
  const fs::path trainLabelsPath{"train-labels-idx1-ubyte"};
  const fs::path testImagesPath{"t10k-images-idx3-ubyte"};
  const fs::path testLabelsPath{"t10k-labels-idx1-ubyte"};
  auto readImgAndLabel = [mnistPath](const fs::path& trainPath,
                                     const fs::path& labelPath) {
    auto absTrainPath = mnistPath / trainPath;
    auto absLabelPath = mnistPath / labelPath;

    std::ifstream imageFile(absTrainPath, std::ios::in | std::ios::binary);
    std::ifstream labelFile(absLabelPath, std::ios::in | std::ios::binary);

    if (!imageFile.is_open() || !labelFile.is_open()) {
      throw std::ios_base::failure("Open image file or label file failed.");
    }

    uint32_t magic, numItems, numLabels, rows, cols;

    imageFile.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    endian_reverse_inplace(magic);
    assert(magic == 2051);
    labelFile.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    endian_reverse_inplace(magic);
    assert(magic == 2049);

    imageFile.read(reinterpret_cast<char*>(&numItems), sizeof(numItems));
    endian_reverse_inplace(numItems);
    labelFile.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
    endian_reverse_inplace(numLabels);
    assert(numLabels == numItems);

    imageFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    endian_reverse_inplace(rows);
    imageFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    endian_reverse_inplace(cols);

    std::cout << "Image and label num is: " << numItems << std::endl;
    std::cout << "Image rows: " << rows << ", cols: " << cols << std::endl;

    std::vector<std::pair<Eigen::Matrix<float, 28, 28, Eigen::RowMajor>,
                          Eigen::RowVectorXf>>
        returnVector;
    Matrix<char, 28, 28, RowMajor> rowMajorNumBuffer;
    char labelTmp;
    for (size_t i = 0; i < numItems; i++) {
      imageFile.read(rowMajorNumBuffer.data(),
                     static_cast<uint64_t>(rows) * cols);
      labelFile.read(&labelTmp, sizeof(labelTmp));
      VectorXf vLavel(10);
      vLavel[labelTmp] = 1.f;
      auto key = rowMajorNumBuffer.cast<float>();
      returnVector.push_back(std::make_pair(key, vLavel));
    }
    return returnVector;
  };

  auto train = readImgAndLabel(trainImagesPath, trainLabelsPath);
  auto test = readImgAndLabel(testImagesPath, testLabelsPath);
  return std::make_tuple(train, test);
}
