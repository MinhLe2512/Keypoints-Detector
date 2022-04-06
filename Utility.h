#pragma once
#ifndef _UTILITY_H_
#define _UTILITY_H_
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\core\mat.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
class Utility{
public:
	float getValueOfMatrix(const cv::Mat& source, int y, int x) {
		int typeMatrix = source.type();
		uchar depth = typeMatrix & CV_MAT_DEPTH_MASK;

		switch (depth) {
			//case CV_8U:  return (float)source.at<uchar>(y, x);
		case CV_32F: return source.at<float>(y, x);
		default:     return (float)source.at<uchar>(y, x);
		}
	}

	bool isLocalMaximaAmongNeighbors(const cv::Mat& source, int y, int x, const std::vector<cv::Mat>& neighbors, int windowSize) {
		int height = source.rows;
		int width = source.cols;
		float* currentValue = new float(getValueOfMatrix(source, y, x));
		for (int i = 0; i < neighbors.size(); i++) {
			if (isLocalMaxima(neighbors[i], y, x, height, width, currentValue, windowSize) == false) {
				return false;
			}
		}

		return isLocalMaxima(source, y, x, height, width, NULL, windowSize);
	}
	bool isLocalMaxima(const cv::Mat& source, int y, int x, int height, int width, float* currentValue, int windowSize) {

		if (currentValue == NULL) {
			currentValue = new float(getValueOfMatrix(source, y, x));
		}

		for (int r = -windowSize / 2; r <= windowSize / 2; r++) {
			for (int c = -windowSize / 2; c <= windowSize / 2; c++) {
				if (y + r < 0 || y + r >= height || x + c < 0 || x + c >= width)
					continue;

				if (getValueOfMatrix(source, y + r, x + c) > * currentValue)
					return false;
			}
		}
		return true;
	}
};


#endif // !_UTILITY_H_

