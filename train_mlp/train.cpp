#include <boost/program_options.hpp>
#include "TrainMlp.h"

int main(int argc, char* const argv[]) {
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")(
      "config", po::value<std::string>(),
      "input the config json file path");
  po::variables_map vm;
  po::store(parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  } else if (vm.count("config")) {
    if (vm["config"].as<std::string>().empty()) {
      std::cerr << "No input json file." << std::endl;
      return 1;
    } else {
      RLDNN::TrainMlp mlp(vm["config"].as<std::string>());
      mlp.startTrain();
    }
  }

  return 0;
}