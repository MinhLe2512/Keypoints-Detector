#include "SiftDetector.h"

double SiftDetector::matchBySIFT(Mat img1, Mat img2, int detector, float threshold) {
	vector<KeyPoint> kp1, kp2;
	Mat gray1, gray2;

	img1.convertTo(gray1, COLOR_BGR2GRAY);
	img2.convertTo(gray2, COLOR_BGR2GRAY);

	switch (detector) {
	//Harris corner detector
	case 1: {
		GlobalDetector gd = GlobalDetector();
		Mat dstImg, cornerMat;
		gd.calculateMoment(gray1);
		gd.detectHarris(cornerMat, 0.05f);

		for (int j = 0; j < cornerMat.rows; j++)
			for (int i = 0; i < cornerMat.cols; i++) {
				//std::cout << corner.at<float>(j, i) << std::endl;
				if (cornerMat.at<float>(j, i) > 160) {
					KeyPoint tmp = KeyPoint(Point(i, j), 4);
					kp1.push_back(tmp);
				}
			}
		GlobalDetector gd2 = GlobalDetector();
		Mat dstImg2, cornerMat2;
		gd2.calculateMoment(gray2);
		gd2.detectHarris(cornerMat2, 0.05f);

		for (int j = 0; j < cornerMat2.rows; j++)
			for (int i = 0; i < cornerMat2.cols; i++) {
				//std::cout << corner.at<float>(j, i) << std::endl;
				if (cornerMat2.at<float>(j, i) > 160) {
					KeyPoint tmp = KeyPoint(Point(i, j), 4);
					kp2.push_back(tmp);
				}
			}
		break;
	}
	//Blob detector with LoG
	case 2: {
		BlobDetector bd = BlobDetector();

		vector<Blob> listBlobs;
		listBlobs = bd.detectBlob_LoG(img1);

		for (int i = 0; i < listBlobs.size(); i++)
			kp1.push_back(KeyPoint(Point(listBlobs[i].x, listBlobs[i].y), listBlobs[i].radius));

		BlobDetector bd2 = BlobDetector();

		listBlobs = bd2.detectBlob_LoG(img2);

		for (int i = 0; i < listBlobs.size(); i++)
			kp2.push_back(KeyPoint(Point(listBlobs[i].x, listBlobs[i].y), listBlobs[i].radius));
		break;
	}
	//Blob detector with DoG
	case 3: {
		BlobDetector bd = BlobDetector();

		vector<Blob> listBlobs;
		listBlobs = bd.detectBlob_DoG(img1);

		for (int i = 0; i < listBlobs.size(); i++)
			kp1.push_back(KeyPoint(Point(listBlobs[i].x, listBlobs[i].y), listBlobs[i].radius));

		BlobDetector bd2 = BlobDetector();

		listBlobs = bd2.detectBlob_DoG(img2);

		for (int i = 0; i < listBlobs.size(); i++)
			kp2.push_back(KeyPoint(Point(listBlobs[i].x, listBlobs[i].y), listBlobs[i].radius));
		break;
	}
	default:
		return 0.0;
		break;
	}
	//Create SIFT Descriptor Extractor
	Ptr<SIFT> sift = SiftDescriptorExtractor::create();
	Mat descriptors1, descriptors2;
	//Compute keypoints
	sift->compute(img1, kp1, descriptors1);
	sift->compute(img2, kp2, descriptors2);

	BFMatcher matcher = BFMatcher();

	vector<vector<DMatch>> matches;
	//Match by KNN
	try {
		matcher.knnMatch(descriptors1, descriptors2, matches, 2);
	}
	catch (Exception err) {
		cout << err.err << endl;
	}

	double distanceSrc;
	double distanceTarget;
	vector<Point2f> srcPoint;
	vector<Point2f> targetPoint;
	vector<DMatch> goodMatch;
	for (size_t i = 0; i < matches.size(); ++i) {
		distanceSrc = matches[i][0].distance;
		distanceTarget = matches[i][1].distance;

		// detect the point that is far from the second one.
		if (distanceSrc <= distanceTarget * threshold) {
			cout << distanceSrc << " " << distanceTarget << endl;
			goodMatch.push_back(matches[i][0]);
			srcPoint.push_back(kp1[matches[i][0].queryIdx].pt);
			targetPoint.push_back(kp2[matches[i][0].trainIdx].pt);
		} 
	}

	// show corresponding point
	Mat drawMatch;
	drawMatches(img1, kp1, img2, kp2, goodMatch, drawMatch,
		Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("DrawMatch", drawMatch);
	waitKey(0);

	switch (detector) {
	case 1:
		imwrite("match_harris.jpg", drawMatch);
		break;
	case 2:
		imwrite("match_log.jpg", drawMatch);
		break;
	case 3:
		imwrite("match_dog.jpg", drawMatch);
		break;
	default:
		return 0.0;
		break;

	}
	
	return 1.0;
}