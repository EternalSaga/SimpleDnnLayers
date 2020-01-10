# simple-two-layers-mlp
A simple two layers multiple layers perceptron based on pure C++.  
I coded it to prove that C++ is also suitable for machine learning / deep learning.  
# Environment
Visual Studio 2019 with C++17  
Update: Support Linux building: with scons and g++ 9.2 or 7.4 (-std=c++17)  
# 3rd party library
Boost 1.71.0: Endian Library, Property Tree Library, Program Options Lirary  
Eigen: For matrix manipulation. I use the master branch at presnet. https://github.com/eigenteam/eigen-git-mirror  
OpenCV: Optional, I only use it for some tests. 
MKL: Optional, if you are not satisfied with the speed of Eigen, you can link it to Eigen.  
# On going
Computational graph is on going.  
Next plan is CNN supporting.  
# Great improvments update
Finally, I gained 6 times speed up than equivalent python version (python3.7 with numpy)