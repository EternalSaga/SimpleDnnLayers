# simple-two-layers-mlp
A simple two layers multiple layers perceptron based on pure C++.

I coded it to prove that C++ is also suitable for machine learning / deep learning.

# Environment
Visual Studio 2019 with C++17
# 3rd party library
Boost 1.71.0: For Endian Library and Property Tree Library

Eigen: For matrix manipulation. I use the master branch at presnet. https://github.com/eigenteam/eigen-git-mirror

OpenCV: Optional, I only use it for some tests.

MKL: Optional, if you are not satisfied with the speed of Eigen, you can link it to Eigen.

# Next Plan
CNN and CUDA support is on going.

# Great improvments update
I changed the eigen version to master branch.
Due to some optimize method which I can't explain it, I gained the 100 times speed improvement.
Finally, I gained 6 times speed up than equivalent python version (python3.7 with numpy)