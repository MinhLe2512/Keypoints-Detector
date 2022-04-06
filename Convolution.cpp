#include "Convolution.h"

float Convolution::getValueOfMatrix(const Mat& source, int y, int x) {
	int typeMatrix = source.type();
	uchar depth = typeMatrix & CV_MAT_DEPTH_MASK;

	switch (depth) {
		//case CV_8U:  return (float)source.at<uchar>(y, x);
	case CV_32F: return source.at<float>(y, x);
	default:     return (float)source.at<uchar>(y, x);
	}
}

void Convolution::setValueOfMatrix(Mat& source, int y, int x, float value) {
	int typeMatrix = source.type();
	uchar depth = typeMatrix & CV_MAT_DEPTH_MASK;

	switch (depth) {
	case CV_32F: source.at<float>(y, x) = value; break;
	default:     source.at<uchar>(y, x) = (uchar)value; break;
	}
}

Mat Convolution::applyOperator(const Mat& a, const Mat& b, string operatorName) {
	int height = a.rows;
	int width = a.cols;
	Mat result = Mat::zeros(height, width, CV_32FC1);

	float (*operatorFunc)(float, float) {};
	if (operatorName == "sum")
		operatorFunc = &sumFunction;
	else if (operatorName == "multiply")
		operatorFunc = &multiplyFunction;
	else if (operatorName == "divide")
		operatorFunc = &divideFunction;
	else if (operatorName == "substract")
		operatorFunc = &substractFuntion;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			setValueOfMatrix(result, y, x, operatorFunc(
				getValueOfMatrix(a, y, x), getValueOfMatrix(b, y, x)
			));
		}
	}

	return result;

}

Mat Convolution::conv2d(const Mat& source, const Mat& kernel) {
	Mat result;
	filter2D(source, result, -1, kernel);
	return result;
}

Mat Convolution::applyGaussianKernel(const Mat& source, float sigmaScale, int kernelSize, float sigma) {
	Mat result = source.clone();
	Mat fSource;
	source.convertTo(fSource, CV_32FC1);
	GaussianBlur(fSource, result, Size(kernelSize, kernelSize), sigmaScale * sigma, 0, BORDER_DEFAULT);
	return result;
}

float Convolution::getMaxValue(const Mat& source) {
	float result = INT_MIN;
	int height = source.rows;
	int width = source.cols;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			result = max(result, getValueOfMatrix(source, y, x));
		}
	}
	return result;
}

bool Convolution::isLocalMaxima(const Mat& source, int y, int x, int height, int width, float* currentValue, int windowSize) {

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

bool Convolution::isLocalMaximaAmongNeighbors(const Mat& source, int y, int x, const vector<Mat>& neighbors, int windowSize) {
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