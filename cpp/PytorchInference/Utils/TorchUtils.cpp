#include "TorchUtils.h"

namespace unetv2 {
	namespace
	{
		// Convert a vector of images to torch Tensor
		std::pair<torch::Tensor, torch::Tensor> convertImageToTensor(const std::pair<cv::Mat, cv::Mat>& images)
		{
			assert(images.first.channels() == images.second.channels());
			assert(images.first.rows == images.second.rows);
			assert(images.first.cols == images.second.cols);
			assert(images.first.type() == images.second.type());
			const int imagesCount = 2;
			const int nChannels = images.first.channels();
			const int height = images.first.rows;
			const int width = images.first.cols;

			const int imageType = images.first.type();

			// Image Type must be one of CV_8U, CV_32F, CV_64F
			assert((imageType % 8 == 0) || ((imageType - 5) % 8 == 0) || ((imageType - 6) % 8 == 0));

			const std::vector<int64_t> dims = { 1, height, width, nChannels };
			const std::vector<int64_t> permute_dims = { 0, 3, 1, 2 };

			std::pair<torch::Tensor, torch::Tensor> imagesAsTensors;
			for (size_t i = 0; i < imagesCount; ++i)
			{
				torch::Tensor imageAsTensor;

				cv::Mat image;
				(i == 0) ? (image = images.first.clone()) : (image = images.second.clone());

				if (imageType % 8 == 0)
				{
					torch::TensorOptions options(torch::kUInt8);
					imageAsTensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
				}
				else if ((imageType - 5) % 8 == 0)
				{
					torch::TensorOptions options(torch::kFloat32);
					imageAsTensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
				}
				else if ((imageType - 6) % 8 == 0)
				{
					torch::TensorOptions options(torch::kFloat64);
					imageAsTensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
				}

				imageAsTensor = imageAsTensor.permute(torch::IntList(permute_dims));
				imageAsTensor = imageAsTensor.toType(torch::kFloat32);
				(i == 0) ? (imagesAsTensors.first = imageAsTensor) : (imagesAsTensors.second = imageAsTensor);
			}

			return imagesAsTensors;
		}

		// Predict
		torch::Tensor Predict(torch::jit::script::Module& model, torch::Tensor tensor)
		{

			std::vector<torch::jit::IValue> inputs;
			inputs.push_back(tensor);

			// Execute the model and turn its output into a tensor.
			torch::NoGradGuard noGrad;
			torch::Tensor output = model.forward(inputs).toTensor();

			torch::DeviceType cpuDeviceType = torch::kCPU;
			torch::Device cpuDevice(cpuDeviceType);
			output = output.to(cpuDevice);

			return output;
		}

		cv::Mat getOutputMask(torch::Tensor& input, const cv::Size origImageSize)
		{
			cv::Mat1f out = cv::Mat1f(origImageSize.height, origImageSize.width, static_cast<float>(0));
			//Remove batch dimension
			input = input.cpu()[0];

			const auto tensor3D = input.accessor<float, 3>();

			for (int i = 0; i < out.rows; ++i)
			{
				float* fptr = out.ptr<float>(i);
				for (int j = 0; j < out.cols; ++j)
				{
					if (tensor3D[0][i][j] <= tensor3D[1][i][j])
					{
						fptr[j] = 255.;
					}
				}
			}
			return out;
		}
	}
	// 1. Read model
	torch::jit::script::Module readModel(const std::string model_path)
	{
		torch::jit::script::Module model = torch::jit::load(model_path);

		torch::DeviceType cpuDeviceType = torch::kCPU;
		torch::Device cpuDevice(cpuDeviceType);

		model.to(cpuDevice);

		return model;
	}

	// 2. Forward
	std::pair<cv::Mat, cv::Mat> forward(const std::pair<cv::Mat, cv::Mat>& images,
		torch::jit::script::Module& model,
		const cv::Size origImageSize)
	{

		// 1. Convert OpenCV matrices to torch Tensor
		std::pair<torch::Tensor, torch::Tensor> tensorPair = convertImageToTensor(images);

		torch::DeviceType cpuDeviceType = torch::kCPU;
		torch::Device cpuDevice(cpuDeviceType);

		tensorPair.first = tensorPair.first.to(cpuDevice);
		tensorPair.second = tensorPair.second.to(cpuDevice);

		// 2. Predict
		torch::Tensor outputFirst = Predict(model, tensorPair.first);
		torch::Tensor outputSecond = Predict(model, tensorPair.second);

		// resize float tensors for beautiful output mask
		namespace F = torch::nn::functional;
		outputFirst = F::interpolate(outputFirst, F::InterpolateFuncOptions()
			.size(std::vector<int64>({ origImageSize.height, origImageSize.width }))
			.mode(torch::kBicubic)
			.align_corners(true));
		outputSecond = F::interpolate(outputSecond, F::InterpolateFuncOptions()
			.size(std::vector<int64>({ origImageSize.height, origImageSize.width }))
			.mode(torch::kBicubic)
			.align_corners(true));

		// 3. Convert torch Tensor to vector of vector of floats
		auto outputMaskFirst = getOutputMask(outputFirst, origImageSize);
		auto outputMaskSecond = getOutputMask(outputSecond, origImageSize);

		return std::make_pair(outputMaskFirst, outputMaskSecond);
	}
}