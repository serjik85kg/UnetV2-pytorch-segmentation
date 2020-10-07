#include "OpencvUtils.h"

namespace unetv2 {
	namespace
	{

		// Resize an image to a given size to
		cv::Mat resizeTo(const cv::Mat& image, const int newHeight, const int newWidth)
		{

			// get original image size
			int orgImageHeight = image.rows;
			int orgImageWidth = image.cols;

			// get image area and resized image area
			float img_area = float(orgImageHeight * orgImageWidth);
			float new_area = float(newHeight * newWidth);

			// resize
			cv::Mat imageScaled;
			cv::Size scale(newWidth, newHeight);

			if (new_area >= img_area)
			{
				cv::resize(image, imageScaled, scale, 0, 0, cv::INTER_LANCZOS4);
			}
			else
			{
				cv::resize(image, imageScaled, scale, 0, 0, cv::INTER_AREA);
			}

			return imageScaled;
		}

		// Normalize an image by subtracting mean and dividing by standard deviation
		cv::Mat normalizeMeanStd(const cv::Mat& image, const std::vector<double> mean, const std::vector<double> std)
		{

			// clone
			cv::Mat imageNormalized = image.clone();

			// convert to float
			imageNormalized.convertTo(imageNormalized, CV_32FC3);

			// subtract mean
			cv::subtract(imageNormalized, mean, imageNormalized);

			// divide by standard deviation
			std::vector<cv::Mat> imgChannels(3);
			cv::split(imageNormalized, imgChannels);

			imgChannels[0] = imgChannels[0] / std[0];
			imgChannels[1] = imgChannels[1] / std[1];
			imgChannels[2] = imgChannels[2] / std[2];

			cv::merge(imgChannels, imageNormalized);

			return imageNormalized;
		}
	}
	// 1. Preprocess
	cv::Mat preprocess(const cv::Mat& image, const int newHeight, const int newWidth,
		const std::vector<double> mean, const std::vector<double> std)
	{

		// Clone
		cv::Mat imageProc = image.clone();

		// Convert from BGR to RGB
		cv::cvtColor(imageProc, imageProc, cv::COLOR_BGR2RGB);

		// Resize image
		imageProc = resizeTo(imageProc, newHeight, newWidth);

		// Convert image to float
		imageProc.convertTo(imageProc, CV_32FC3);

		// 3. Normalize to [0, 1]
		imageProc = imageProc / 255.0;

		// 4. Subtract mean and divide by std
		imageProc = normalizeMeanStd(imageProc, mean, std);

		return imageProc;
	}

}