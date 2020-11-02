enable_testing()
file(GLOB_RECURSE LayerTestSrc "*.hpp" "*.cpp")
add_executable(LayerTest ${LayerTestSrc})
target_include_directories(LayerTest PRIVATE "/home/robin/cpp_proj/simple-two-layers-mlp/thirdparty/eigen/Eigen" "../Layers")
find_package(Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
target_link_libraries(LayerTest PRIVATE
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                      NonOptCLib
                      )
target_compile_features(LayerTest PRIVATE cxx_std_17)