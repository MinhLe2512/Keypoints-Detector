#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "GlobalDetector.h"
#include "BlobDetector.h"
#include "SiftDetector.h"

using namespace cv;

int main(int argc, char* argv[]) {
	//Input/ Training image
	char* inFile = new char[20];
	//Target image
	char* inFile2 = new char[20];
	char* flag = new char[20];
	int parameter;

	if (argc == 4) {
		for (int i = 1; i < argc; i++) {
			if (i == 1) inFile = argv[i];
			else if (i == 2) flag = argv[i];
			else if (i == 3) parameter = atoi(argv[i]);
		}
	}
	else if (argc == 5) {
		for (int i = 1; i < argc; i++) {
			if (i == 1) inFile = argv[i];
			else if (i == 2) inFile2 = argv[i];
			else if (i == 3) flag = argv[i];
			else if (i == 4) parameter = atoi(argv[i]);
		}
	}

	cv::Mat srcImg = cv::imread(inFile, cv::IMREAD_COLOR);
	cv::Mat targetImg = cv::imread(inFile2, cv::IMREAD_COLOR);

	//cv::imshow("Help", srcImg);
	cv::Mat grayImg;
	cvtColor(srcImg, grayImg, cv::COLOR_BGR2GRAY);

	//Harris corners detector
	if (strcmp(flag, "-harris") == 0) {
		//Mat corners_cv = Mat::zeros(grayImg.size(), CV_32FC1);
		//cornerHarris(grayImg, corners_cv, 3, 3, 0.05, BORDER_DEFAULT);

		//// --- normalize and save
		//Mat corners_norm_cv;
		//normalize(corners_cv, corners_norm_cv, 0, 255, NORM_MINMAX, CV_32FC1);
		//Mat dst_norm_scaled_cv;
		//convertScaleAbs(corners_norm_cv, dst_norm_scaled_cv);

		//// --- Drawing a circle around corners and save
		//GlobalDetector gDetect2 = GlobalDetector();
		//Mat img_corners_cv;
		//int thresh = 140;       // Threshold for corners
		//gDetect2.drawCorners(srcImg, img_corners_cv, corners_norm_cv, thresh, Scalar(255, 0, 0));
		//imshow("corners on image by opencv", img_corners_cv);
		//imwrite("img_corners_cv.png", img_corners_cv);


		cv::Mat dstImg, cornerMat;

		GlobalDetector gDetect = GlobalDetector();
		gDetect.calculateMoment(grayImg);
		gDetect.detectHarris(cornerMat, 0.05f);
		//cv::convertScaleAbs(cornerMat, cornerMat);

		gDetect.drawCorners(srcImg, dstImg, cornerMat, 160, cv::Scalar(0, 0, 255));
		cv::imshow("Output image", dstImg);
		cv::imwrite("harris_detector.png", dstImg);
	}
	//Blob detector with LoG
	else if (strcmp(flag, "-LoG") == 0) {
		cv::Mat dstImg;
		BlobDetector bl;
		bl.detectBlob_LoG(srcImg);
	}
	//Blob detector with DoG
	else if (strcmp(flag, "-DoG") == 0) {
		cv::Mat dstImg;
		BlobDetector bl;
		bl.detectBlob_DoG(srcImg);
	}
	//Match image using key points from Harris
	else if (strcmp(flag, "-matchByHarris") == 0) {
		SiftDetector sf = SiftDetector();
		sf.matchBySIFT(srcImg, targetImg, 1, 0.85);
	}
	//Match image using key points from LoG
	else if (strcmp(flag, "-matchByLoG") == 0) {
		SiftDetector sf = SiftDetector();
		sf.matchBySIFT(srcImg, targetImg, 2, 0.8);
	}
	//Match image using key points from DoG
	else if (strcmp(flag, "-matchByDoG") == 0) {
		SiftDetector sf = SiftDetector();
		sf.matchBySIFT(srcImg, targetImg, 3, 0.75);
	}
	cv::waitKey(0);
	return 1;
}