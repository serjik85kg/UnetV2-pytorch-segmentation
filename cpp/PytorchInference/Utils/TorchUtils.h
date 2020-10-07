#pragma once

#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <memory>

#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>
#include <torch/nn/functional.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace unetv2 {

	torch::jit::script::Module readModel(const std::string modelPath);

	std::pair<cv::Mat, cv::Mat> forward(const std::pair<cv::Mat, cv::Mat>& inputPair,
		torch::jit::script::Module& torchModule, const cv::Size origImageSize);
}
