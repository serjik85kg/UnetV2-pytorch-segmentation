#pragma once

#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

cv::Mat preprocess( const cv::Mat& inputImage, const int newHeight, const int newWidth,
					const std::vector<double> mean,
					const std::vector<double> std );
