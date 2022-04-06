#include "GlobalDetector.h"

void GlobalDetector::drawCorners(cv::Mat srcImg, cv::Mat& dstImg, cv::Mat corner, int thresh, cv::Scalar color)
{
	srcImg.copyTo(dstImg);
	for (int j = 0; j < corner.rows; j++)
		for (int i = 0; i < corner.cols; i++) {
			//std::cout << corner.at<float>(j, i) << std::endl;
			if (corner.at<float>(j, i) > thresh)
				cv::circle(dstImg, cv::Point(i, j), 4, color, 1, cv::LINE_4, 0);
		}
}


int GlobalDetector::detectHarris(cv::Mat& dstImg, float k) {
	dstImg = cv::Mat::zeros(IxIx.rows, IxIx.cols, CV_32F);
	float a, b, c;

	//Gaussian Blur
	GaussianBlur::GaussianBlur(IxIx, IxIx, 5, 1.0);
	GaussianBlur::GaussianBlur(IxIy, IxIy, 5, 1.0);
	GaussianBlur::GaussianBlur(IyIy, IyIy, 5, 1.0);

	//Find corners
	for (int i = 1; i < IxIx.cols - 1; i++) {
		for (int j = 1; j < IxIx.rows - 1; j++) {
			a = IxIx.at<float>(j, i);
			b = IxIy.at<float>(j, i);
			c = IyIy.at<float>(j, i);
			//std::cout << IxIx.at<float>(j, i) << std::endl;
			//Calculate 
			dstImg.at<float>(j, i) = a * c - b * b / 4 - k * (a + c) * (a + c);
			//std::cout << dstImg.at<float>(j, i) << std::endl;
		}
	}
	//Normalize
	cv::normalize(dstImg, dstImg, 0, 255, cv::NORM_MINMAX);
	return 1;
}

void GlobalDetector::calculateMoment(cv::Mat srcImg) {
	int rows = srcImg.rows, cols = srcImg.cols;
	srcImg.convertTo(srcImg, CV_32F);

	std::setprecision(4);
	IxIx = cv::Mat::zeros(rows, cols, CV_32F);
	IyIy = cv::Mat::zeros(rows, cols, CV_32F);
	IxIy = cv::Mat::zeros(rows, cols, CV_32F);
	
	//Step 2: Spatial derivative calculation
	float kernelX[3][3] = { { -1,-2,-1 },
					   {  0, 0, 0 },
					   {  1, 2, 1 } };
	float kernelY[3][3] = { { -1, 0, 1 },
					   { -2, 0, 2 },
					   { -1, 0, 1 } }; 

	//Derivative of X and Y
	float sumX = 0.0, sumY = 0.0;
	for (int i = 1; i < cols - 1; i++) {
		for (int j = 1; j < rows - 1; j++) {
			sumX = sumY = 0.0;
			for (int ii = -1; ii <= 1; ii++) {
				for (int jj = -1; jj <= 1; jj++) {
					sumX += kernelX[jj + 1][ii + 1] * srcImg.at<float>(j + jj, i + ii);
					sumY += kernelY[jj + 1][ii + 1] * srcImg.at<float>(j + jj, i + ii);
				}
			}
			sumX = abs(sumX);
			sumY = abs(sumY);
			IxIx.at<float>(j, i) = sumX * sumX;
			IxIy.at<float>(j, i) = sumX * sumY;
			IyIy.at<float>(j, i) = sumY * sumY;
		}
	}
}

GaussianBlur::GaussianBlur(const cv::Mat& srcImg, cv::Mat& dstImg, const int kernel_size, const double sigma) {
	std::vector<double> kernel = getKernel(kernel_size, sigma);

	unsigned char* data_in = (unsigned char*)(srcImg.data);
	unsigned char* data_out = (unsigned char*)(dstImg.data);

	for (int row = 0; row < srcImg.rows; row++) {
		for (int col = 0; col < (srcImg.cols * srcImg.channels()); col += srcImg.channels()) {
			for (int channel_index = 0; channel_index < srcImg.channels(); channel_index++) {

				if (row <= kernel_size / 2 || row >= srcImg.rows - kernel_size / 2 ||
					(srcImg.cols * srcImg.channels()) <= kernel_size / 2 ||
					col >= (srcImg.cols * srcImg.channels()) - kernel_size / 2) {
					data_out[dstImg.step * row + col + channel_index] = data_in[dstImg.step * row + col + channel_index];
					continue;
				}

				int k_ind = 0;
				double sum = 0;
				for (int k_row = -kernel_size / 2; k_row <= kernel_size / 2; ++k_row) {
					for (int k_col = -kernel_size / 2; k_col <= kernel_size / 2; ++k_col) {
						sum += kernel[k_ind] * (data_in[srcImg.step * (row + k_row) + col + (k_col * srcImg.channels()) + channel_index]);
						//std::cout << sum << std::endl;
						k_ind++;
					}
				}
				data_out[dstImg.step * row + col + channel_index] = (unsigned int)(std::max(std::min(sum, 1.0), 0.0));
				//std::cout << (unsigned int)(std::max(std::min(sum, 1.0), 0.0) * 255.0) << std::endl;
			}
		}
	}
}
std::vector<double> GaussianBlur::getKernel(const int kernel_size, double sigma)
{
	std::setprecision(8);
	std::vector<double> kernel(kernel_size * kernel_size, 0);

	if (sigma <= 0) {
		sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8;
	}
	double r, s = 2.0 * sigma * sigma;

	// sum is for normalization
	double sum = 0.0;

	// generating nxn kernel
	int i, j;
	double mean = kernel_size / 2;
	for (i = 0; i < kernel_size; i++) {
		for (j = 0; j < kernel_size; j++) {
			kernel[(i * kernel_size) + j] = exp(-0.5 * (pow((i - mean) / sigma, 2.0) + pow((j - mean) / sigma, 2.0)))
				/ (2 * 3.14f * sigma * sigma);
			sum += kernel[(i * kernel_size) + j];
		}
	}

	// normalising the Kernel
	for (int i = 0; i < kernel.size(); ++i) {
		kernel[i] /= sum;
		//std::cout << kernel[i] << std::endl;
	}

	return kernel;
}
