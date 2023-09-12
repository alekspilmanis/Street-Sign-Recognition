#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/* 
*  This program is only used for development. Through the use of the test image, and the
*  sliders, all the hsv mask values can be found for each of the sign colors, so that each
*  mask only detects one kind of sign. These values are hardcoded into classify.cpp.
*/

Mat imgHSV, mask;
int hmin = 0, smin = 110, vmin = 153;
int hmax = 19, smax = 240, vmax = 255;

int main() {

	string path = "test.png";
	Mat img = imread(path);

	Mat imgResize;

	resize(img, imgResize, Size(1050, 750));


	cvtColor(imgResize, imgHSV, COLOR_BGR2HSV);

	namedWindow("Trackbars", (640, 200));
	createTrackbar("Hue Min", "Trackbars", &hmin, 179);
	createTrackbar("Hue Max", "Trackbars", &hmax, 179);
	createTrackbar("Sat Min", "Trackbars", &smin, 255);
	createTrackbar("Sat Max", "Trackbars", &smax, 255);
	createTrackbar("Val Min", "Trackbars", &vmin, 255);
	createTrackbar("Val Max", "Trackbars", &vmax, 255);

	while (true) {

		Scalar lower(hmin, smin, vmin);
		Scalar upper(hmax, smax, vmax);


		inRange(imgHSV, lower, upper, mask);

		imshow("Image", imgResize);
		imshow("ImgHSV", imgHSV);
		imshow("ImageHSV", mask);
		waitKey(1);
	}

	return 0;
}