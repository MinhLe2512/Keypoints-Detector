#pragma once
#ifndef _BLOB_DETECTOR_H_
#define _BLOB_DETECTOR_H_

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\core\mat.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include "Utility.h"
#include "Convolution.h"

struct Blob {
    int x;
    int y;
    int radius;
};


class BlobDetector {
public:
    static const string result_dir;

private:
    static const float k;


public:
     vector<Blob> detectBlob_LoG(const Mat& source, float startSigma = 1.0, int nLayers = 10);
     vector<Blob> detectBlob_DoG(const Mat& source, float startSigma = 1.0, int nLayers = 8);

     Mat visualizeResult(const Mat& source, vector<Blob> blobs, int k);
private:
    // helper function for detectBlob_LoG
     vector<Mat> getScaleLaplacianImages(const Mat& source, vector<float>& maxLogValues, float startSigma = 1.0, int nLayers = 10);

    // helper function for detectBlob_DoG
     vector<Mat> getScaleLaplacianImages_DoG(const Mat& source, vector<float>& maxLogValues, float startSigma = 1.0, int nLayers = 8);

     vector<Blob> getLocalMaximumPoints(vector<Mat> listLogImages, const vector<float>& maxLogValues, float logThres = 0.05, float startSigma = 1.0);
     Mat createLoGkernel(int kernelSize, float kernelSigma);

     int getLogFilterSize(float sigma);

     Mat calculateLoG(const Mat& source, float sigma);
     Mat calculateGaussian(const Mat& source, float sigma);

};

#endif // !_BLOB_DETECTOR_H_