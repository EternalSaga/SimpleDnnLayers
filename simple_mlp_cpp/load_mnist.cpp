#include "load_mnist.h"
namespace RLDNN {
std::tuple<MNistData, MNistData> loadMnist(fs::path mnistPath,
                                           bool normalize,
                                           bool flatten) {
  using namespace Eigen;
  using boost::endian::endian_reverse_inplace;
  const fs::path trainImagesPath{"train-images-idx3-ubyte"};
  const fs::path trainLabelsPath{"train-labels-idx1-ubyte"};
  const fs::path testImagesPath{"t10k-images-idx3-ubyte"};
  const fs::path testLabelsPath{"t10k-labels-idx1-ubyte"};
  auto readImgAndLabel = [mnistPath, normalize](const fs::path& trainPath,
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

    Eigen::Matrix<unsigned char, Eigen::Dynamic, MNIST_LENGTH, Eigen::RowMajor>
        allMnistChar(numItems, MNIST_LENGTH);
    allMnistChar.setZero();
    LabelData allLabels(numItems, LABEL_LENGTH);
    allLabels.setZero();
    char labelTmp;

    for (size_t i = 0; i < numItems; i++) {
      imageFile.read(reinterpret_cast<char*>(allMnistChar.row(i).data()),
                     static_cast<uint64_t>(MNIST_LENGTH));
      labelFile.read(&labelTmp, sizeof(labelTmp));
      allLabels.row(i)(labelTmp) = 1.f;
    }
    Eigen::Matrix<float, Eigen::Dynamic, MNIST_LENGTH, Eigen::RowMajor>
        allMnist = allMnistChar.cast<float>();
    if (normalize) {
      allMnist /= 255.0f;
    }
    return std::make_pair(allMnist, allLabels);
  };

  auto train{readImgAndLabel(trainImagesPath, trainLabelsPath)};
  auto test{readImgAndLabel(testImagesPath, testLabelsPath)};
  return std::make_tuple(train, test);
}
}  // namespace RLDNN