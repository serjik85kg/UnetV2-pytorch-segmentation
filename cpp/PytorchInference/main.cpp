#include "Utils\OpencvUtils.h"
#include "Utils\TorchUtils.h"

cv::Mat addMask( cv::Mat imageBGR, cv::Mat1f mask, cv::Scalar color )
{
	cv::Mat output;
	cv::Mat maskConv;
	mask.convertTo( maskConv, CV_8UC3 );
	std::vector<cv::Mat> maskVec{maskConv, maskConv, maskConv};
	cv::merge( maskVec, maskConv );

	for ( size_t i = 0; i < maskConv.rows; ++i )
	{
		cv::Vec3b* rowPtr = maskConv.ptr<cv::Vec3b>( i );
		for ( size_t j = 0; j < maskConv.cols; ++j )
		{
			if ( rowPtr[j][0] == 255 )
				rowPtr[j] = cv::Vec3b( color[0], color[1], color[2] );
		}
	}

	cv::addWeighted( imageBGR, 1.0, maskConv, 0.5, 0, output );
	return output;
}

cv::Scalar redMask = cv::Scalar( 0, 0, 255 );
cv::Scalar greenMask = cv::Scalar( 0, 255, 0 );
cv::Scalar blueMask = cv::Scalar( 255, 0, 0 );
cv::Scalar yellowMask = cv::Scalar( 0, 255, 255 );
cv::Scalar purpleMask = cv::Scalar( 255, 0, 255 );
cv::Scalar lightblueMask = cv::Scalar( 255, 255, 0 );

// RAW main, edit it in future
int main( )
{
	std::string modelPath = "../../python/test_outputs/traced_model_unet.pt";
	std::string imagePath1 = "examples/42f.jpg";
	std::string imagePath2 = "examples/42p.jpg";
	const auto imgHeight = 256;
	const auto imgWidth = 256;
	const std::vector<double> mean = {0.485, 0.456, 0.406};
	const std::vector<double> std = {0.229, 0.224, 0.225};

	cv::Mat img1 = cv::imread( imagePath1 );
	cv::Mat img2 = cv::imread( imagePath2 );
	const cv::Size origSize = img1.size( );
	assert( img1.size( ) == img2.size( ) );

	const cv::Mat outputImg1 = img1.clone( );
	const cv::Mat outputImg2 = img2.clone( );

	img1 = unetv2::preprocess( img1, imgHeight, imgWidth, mean, std );
	img2 = unetv2::preprocess( img2, imgHeight, imgWidth, mean, std );

	auto torchModel = unetv2::readModel( modelPath );

	const auto inputPair = std::make_pair( img1, img2 );
	const auto out = unetv2::forward( inputPair, torchModel, origSize );

	auto output1 = addMask( outputImg1, out.first, purpleMask );
	auto output2 = addMask( outputImg2, out.second, redMask );

	cv::resize(output1, output1, { output1.cols / 2, output1.rows / 2 });
	cv::resize(output2, output2, { output2.cols / 2, output2.rows / 2 });

	cv::imshow( "1", output1 );
	cv::imshow( "2", output2 );

	//cv::imshow( "out.first", out.first );
	//cv::imshow( "out.second", out.second );

	cv::waitKey( );
}