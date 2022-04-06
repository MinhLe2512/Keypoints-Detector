#include "BlobDetector.h"

const float BlobDetector::k = sqrt(2);
const string BlobDetector::result_dir = "./result/blob";

 Mat BlobDetector::createLoGkernel(int kernelSize, float kernelSigma) {
    Mat logKernel = Mat::zeros(kernelSize, kernelSize, CV_32FC1);
    float sum = 0.0;
    float var = 2 * kernelSigma * kernelSigma;
    float r;

    for (int y = -(kernelSize / 2); y <= kernelSize / 2; y++) {
        for (int x = -(kernelSize / 2); x <= kernelSize / 2; x++) {
            r = sqrt(x * x + y * y);
            float xySigma = (float)(x * x + y * y) / (2 * kernelSigma * kernelSigma);
            float value = (1.0 / (3.14 * pow(kernelSigma, 4))) * (1 - xySigma) * exp(-xySigma);
            logKernel.at<float>(y + kernelSize / 2, x + kernelSize / 2) = value;
            sum += logKernel.at<float>(y + kernelSize / 2, x + kernelSize / 2);
        }
    }

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            logKernel.at<float>(i, j) /= sum;
        }
    }

    return logKernel;
}

vector<Blob> BlobDetector::detectBlob_LoG(const Mat& source, float startSigma, int nLayers) {
    //convert to gray image and remove noise
    Mat grayImage;
    cvtColor(source, grayImage, COLOR_BGR2GRAY);
    Mat smoothenSource = Convolution::applyGaussianKernel(grayImage, 1.2);

    //1. generate scale-normalized LoG kernels and use them to filter image
    vector<float> maxLogValues;
    vector<Mat> logImages = getScaleLaplacianImages(smoothenSource, maxLogValues, startSigma, nLayers);

    //2. find local maximum points
    vector<Blob> blobs = getLocalMaximumPoints(logImages, maxLogValues);

    //3. visualize result
    visualizeResult(source, blobs, 1);

    return blobs;
}

vector<Blob> BlobDetector::detectBlob_DoG(const Mat& source, float startSigma, int nLayers) {
    //convert to gray image and remove noise
    Mat grayImage;
    cvtColor(source, grayImage, COLOR_BGR2GRAY);
    Mat smoothenSource = Convolution::applyGaussianKernel(grayImage, 1);

    //1. generate scale-normalized DoG kernels and use them to filter image
    vector<float> maxLogValues;
    vector<Mat> logImages = getScaleLaplacianImages_DoG(smoothenSource, maxLogValues, startSigma, nLayers);

    //2. find local maximum points
    vector<Blob> blobs = getLocalMaximumPoints(logImages, maxLogValues);

    //3. visualize result
    visualizeResult(source, blobs, 2);

    return blobs;
}

vector<Mat> BlobDetector::getScaleLaplacianImages(const Mat& source, vector<float>& maxLogValues, float startSigma, int nLayers) {
    vector<Mat> logImages;
    float sigma = startSigma;

    for (int i = 1; i <= nLayers; i++) {
        float scaledSigma = sigma * pow(k, i);

        Mat logImage = calculateLoG(source, scaledSigma);

        // square log image
        logImage = Convolution::applyOperator(logImage, logImage, "multiply");
        maxLogValues.push_back(Convolution::getMaxValue(logImage));
        logImages.push_back(logImage);
    }

    return logImages;
}

vector<Mat> BlobDetector::getScaleLaplacianImages_DoG(const Mat& source, vector<float>& maxLogValues, float startSigma, int nLayers) {
    vector<Mat> dogImages;
    float sigma = startSigma;

    Mat prevGauss = calculateGaussian(source, startSigma);

    for (int i = 1; i <= nLayers; i++) {
        float scaledSigma = sigma * pow(k, i);
        Mat curGauss = calculateGaussian(source, scaledSigma);

        // square dog image
        Mat dogImage = Convolution::applyOperator(curGauss, prevGauss, "substract");
        Mat squareDogImage = Convolution::applyOperator(dogImage, dogImage, "multiply");

        maxLogValues.push_back(Convolution::getMaxValue(squareDogImage));
        dogImages.push_back(squareDogImage);

        prevGauss = curGauss;
    }

    return dogImages;
}

vector<Blob> BlobDetector::getLocalMaximumPoints(vector<Mat> listLogImages, const vector<float>& maxLogValues, float logThres, float startSigma) {
    Mat firstImage = listLogImages[0];
    int height = firstImage.rows;
    int width = firstImage.cols;
    int nImages = listLogImages.size();

    vector<Blob> candidates;

    for (int i = 1; i < nImages - 1; i++) {
        vector<Mat> neighbors;
        if (i == 0) {
            neighbors.push_back(listLogImages[i + 1]);
        }
        else if (i == nImages - 1) {
            neighbors.push_back(listLogImages[i - 1]);
        }
        else {
            neighbors.push_back(listLogImages[i + 1]);
            neighbors.push_back(listLogImages[i - 1]);
        }
        int layerBLob = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (Convolution::isLocalMaximaAmongNeighbors(listLogImages[i], y, x, neighbors) &&
                    Convolution::getValueOfMatrix(listLogImages[i], y, x) > logThres * maxLogValues[i]) {

                    float radius = pow(k, i + 1) * sqrt(2) * startSigma;
                    candidates.push_back({ x, y, (int)radius });
                    layerBLob++;
                }
            }
        }

        cout << "choose " << layerBLob << " from layer " << i << " radius: " << pow(k, i + 1) * sqrt(2) * startSigma << endl;
    }
    return candidates;
}

Mat BlobDetector::visualizeResult(const Mat& source, vector<Blob> blobs, int k) {
    Mat copyImage = source.clone();

    for (int i = 0; i < blobs.size(); i++) {
        circle(copyImage, Point(blobs[i].x, blobs[i].y), blobs[i].radius, Scalar(0, 0, 255), 1);
    }

    imshow("blob result", copyImage);
    switch (k) {
    case 1:
        imwrite("LoG_image.jpg", copyImage);
        break;
    case 2:
        imwrite("DoG_image.jpg", copyImage);
        break;
    }
    waitKey(0);
    return copyImage;
}

int BlobDetector::getLogFilterSize(float sigma) {
    int filterSize = 2 * ceil(3 * sigma) + 1;
    return filterSize;//(int)(sigma+5);
}


Mat BlobDetector::calculateLoG(const Mat& source, float sigma) {
    int kernelSize = getLogFilterSize(sigma);
    Mat logKernel = createLoGkernel(kernelSize, sigma);
    Mat logImage = Convolution::conv2d(source, logKernel);

    return logImage;
}

Mat BlobDetector::calculateGaussian(const Mat& source, float sigma) {
    int kernelSize = getLogFilterSize(sigma);
    Mat gaussImage = Convolution::applyGaussianKernel(source, 1, kernelSize, sigma);
    return gaussImage;
}