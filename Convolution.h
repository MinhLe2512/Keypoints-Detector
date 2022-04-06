#pragma once
#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_
#include <opencv2\core\mat.hpp>
#include <vector>
#include <iostream>
#include "common.h"
using namespace cv;
using namespace std;

class Convolution
{
public:
    static Mat applyGaussianKernel(const Mat& source, float sigmaScale, int kernelSize = 3, float sigma = 1.0);
    static Mat conv2d(const Mat& source, const Mat& kernel);

    static void setValueOfMatrix(Mat& source, int y, int x, float value);
    static float getValueOfMatrix(const Mat& source, int y, int x);
    static float getMaxValue(const Mat& source);
    static Mat applyOperator(const Mat& a, const Mat& b, string operatorName);

    static bool isLocalMaxima(const Mat& source, int y, int x, int height, int width, float* currentValue = NULL, int windowSize = 3);
    static bool isLocalMaximaAmongNeighbors(const Mat& source, int y, int x, const vector<Mat>& neighbors, int windowSize = 3);
};
#endif // ! _CONVOLUTION_H_