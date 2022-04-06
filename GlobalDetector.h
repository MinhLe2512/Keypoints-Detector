#pragma once
#ifndef _GLOABL_DETECTOR_H_
#define _GLOABL_DETECTOR_H_

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\core\mat.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include "Utility.h"
#include "Convolution.h"

class GlobalDetector {
private:
	cv::Mat IxIx;
	cv::Mat IyIy;
	cv::Mat IxIy;
public:
	int detectHarris(cv::Mat& dstImg, float k);
	void calculateMoment(const cv::Mat srcImg);
	void drawCorners(cv::Mat srcImg, cv::Mat& dstImg, cv::Mat corner, int thresh, cv::Scalar color);
	~GlobalDetector() {
		IxIx.release();
		IxIy.release();
		IyIy.release();
	}
};



class GaussianBlur {
public:
	GaussianBlur(const cv::Mat& srcImg, cv::Mat& dstImg, const int kernel_size, const double sigma);
	std::vector<double> getKernel(const int kernel_size, double sigma);
};
#endif // !_GLOABL_DETECTOR_H_
