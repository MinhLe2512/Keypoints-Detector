#pragma once
#ifndef _SIFT_DETECTOR_H_
#define _SIFT_DETECTOR_H_

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\core\mat.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include "Utility.h"
#include "Convolution.h"
#include "GlobalDetector.h"
#include "BlobDetector.h"

using namespace std;
using namespace cv;

class SiftDetector {
public:
	double matchBySIFT(Mat img1, Mat img2, int detector, float threshold);
};
#endif 