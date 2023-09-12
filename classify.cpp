#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


// The amount of declarations here is due to the fact that a seperate mask is needed for each sign color
// For more information on how the HSV values for each mask were found, see find_masks.cpp

Mat imgHSV;

// Red, for stop signs and yield signs
Mat maskR, img_cannyR, img_blurR, img_dilR;
int hminR = 172, sminR = 177, vminR = 189;
int hmaxR = 179, smaxR = 255, vmaxR = 255;
Scalar lowerR(hminR, sminR, vminR);
Scalar upperR(hmaxR, smaxR, vmaxR);

// Neon Green, for Ped Xing
Mat maskNG, img_cannyNG, img_blurNG, img_dilNG;
int hminNG = 26, sminNG = 229, vminNG = 177;
int hmaxNG = 43, smaxNG = 255, vmaxNG = 255;
Scalar lowerNG(hminNG, sminNG, vminNG);
Scalar upperNG(hmaxNG, smaxNG, vmaxNG);

// Yellow, for Warning / Advisory Signs
Mat maskY, img_cannyY, img_blurY, img_dilY;
int hminY = 18, sminY = 205, vminY = 170;
int hmaxY = 26, smaxY = 255, vmaxY = 219;
Scalar lowerY(hminY, sminY, vminY);
Scalar upperY(hmaxY, smaxY, vmaxY);

// Orange, for Road Work
Mat maskO, img_cannyO, img_blurO, img_dilO;
int hminO = 0, sminO = 219, vminO = 180;
int hmaxO = 12, smaxO = 250, vmaxO = 255;
Scalar lowerO(hminO, sminO, vminO);
Scalar upperO(hmaxO, smaxO, vmaxO);

// Green, for Interchange / Street signs
Mat maskG, img_cannyG, img_blurG, img_dilG;
int hminG = 79, sminG= 199, vminG = 76;
int hmaxG = 104, smaxG = 255, vmaxG = 123;
Scalar lowerG(hminG, sminG, vminG);
Scalar upperG(hmaxG, smaxG, vmaxG);

int main() {

	// Reads user input from upload folder
	string path = "image.png";
	Mat input_img = imread(path);

	// Convert user image into HSV
	cvtColor(input_img, imgHSV, COLOR_BGR2HSV);

	// Create mask for each sign color
	inRange(imgHSV, lowerR, upperR, maskR);
	inRange(imgHSV, lowerNG, upperNG, maskNG);
	inRange(imgHSV, lowerY, upperY, maskY);
	inRange(imgHSV, lowerO, upperO, maskO);
	inRange(imgHSV, lowerG, upperG, maskG);

	// Apply some post-processing to each mask to help with contour detection
	GaussianBlur(maskR, img_blurR, Size(3, 3), 3, 0);
	Canny(img_blurR, img_cannyR, 25, 75);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(img_cannyR, img_dilR, kernel); 

	GaussianBlur(maskNG, img_blurNG, Size(3, 3), 3, 0);
	Canny(img_blurNG, img_cannyNG, 25, 75);
	dilate(img_cannyNG, img_dilNG, kernel);

	GaussianBlur(maskY, img_blurY, Size(3, 3), 3, 0);
	Canny(img_blurY, img_cannyY, 25, 75);
	dilate(img_cannyY, img_dilY, kernel);

	GaussianBlur(maskO, img_blurO, Size(3, 3), 3, 0);
	Canny(img_blurO, img_cannyO, 25, 75);
	dilate(img_cannyO, img_dilO, kernel);

	GaussianBlur(maskG, img_blurG, Size(3, 3), 3, 0);
	Canny(img_blurG, img_cannyG, 25, 75);
	dilate(img_cannyG, img_dilG, kernel);

	/* 
	*  At this point, each mask is fully post-processed and stored in their respective img_dil variables.
	*  Now it is time to find the contours, approximate polygons, and in the case of Red, differentiate 
	*  between a stop sign and a yield sign by the number of conrners. Lasty, the bounding rectangles and
	*  sign type are drawn on the original image.
	*/

	// For Red
	vector<vector<Point>> contoursR;
	vector<Vec4i> hierarchyR;

	findContours(img_dilR, contoursR, hierarchyR, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> conPolyR(contoursR.size());
	vector<Rect> boundRectR(contoursR.size());

	// Only need "signType" for Red, since there are two red signs, yield and stop.
	String signType;

	// Itterate through all red polygons detected in the image, bound and write text for each.

	for (int i = 0; i < contoursR.size(); i++) {

		int areaR = contourArea(contoursR[i]);

		// 1000 is an arbitrary number, it just helps with noise filtration
		if (areaR > 1000) {

			float periR = arcLength(contoursR[i], true);
			approxPolyDP(contoursR[i], conPolyR[i], 0.02 * periR, true);
			drawContours(input_img, conPolyR, i, Scalar(255, 0, 255), 2);
			boundRectR[i] = boundingRect(conPolyR[i]);
			rectangle(input_img, boundRectR[i].tl(), boundRectR[i].br(), Scalar(0, 225, 0), 5);

			// Determine number of corners
			int objCor = (int)conPolyR[i].size();

			if (objCor == 3) { signType = "Yield Sign"; }
			else { signType = "Stop Sign"; }

			// Write sign type near bounding rect
			putText(input_img, signType, { boundRectR[i].x, boundRectR[i].y - 5, }, FONT_HERSHEY_PLAIN, 1, Scalar(0, 69, 255), 2);

		}
	}

	// For Neon Green
	vector<vector<Point>> contoursNG;
	vector<Vec4i> hierarchyNG;

	findContours(img_dilNG, contoursNG, hierarchyNG, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> conPolyNG(contoursNG.size());
	vector<Rect> boundRectNG(contoursNG.size());

	// Itterate through all Neon Green polygons detected in the image, bound and write text for each.
	for (int i = 0; i < contoursNG.size(); i++) {

		int areaNG = contourArea(contoursNG[i]);

		if (areaNG > 1000) {

			float periNG = arcLength(contoursNG[i], true);
			approxPolyDP(contoursNG[i], conPolyNG[i], 0.02 * periNG, true);
			drawContours(input_img, conPolyNG, i, Scalar(255, 0, 255), 2);
			boundRectNG[i] = boundingRect(conPolyNG[i]);
			rectangle(input_img, boundRectNG[i].tl(), boundRectNG[i].br(), Scalar(0, 225, 0), 5);

			// Write sign type near bounding rect
			putText(input_img, "Pedestrian Crosswalk Sign", { boundRectNG[i].x, boundRectNG[i].y - 5, }, FONT_HERSHEY_PLAIN, 1, Scalar(0, 69, 255), 2);
		}
	}

	// For Yellow
	vector<vector<Point>> contoursY;
	vector<Vec4i> hierarchyY;

	findContours(img_dilY, contoursY, hierarchyY, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> conPolyY(contoursY.size());
	vector<Rect> boundRectY(contoursY.size());

	// Itterate through all Yellow polygons detected in the image, bound and write text for each.
	for (int i = 0; i < contoursY.size(); i++) {

		int areaY = contourArea(contoursY[i]);

		if (areaY > 1000) {

			float periY = arcLength(contoursY[i], true);
			approxPolyDP(contoursY[i], conPolyY[i], 0.02 * periY, true);
			drawContours(input_img, conPolyY, i, Scalar(255, 0, 255), 2);
			boundRectY[i] = boundingRect(conPolyY[i]);
			rectangle(input_img, boundRectY[i].tl(), boundRectY[i].br(), Scalar(0, 225, 0), 5);

			// Write sign type near bounding rect
			putText(input_img, "Warning / Advisory Sign", { boundRectY[i].x, boundRectY[i].y - 5, }, FONT_HERSHEY_PLAIN, 1, Scalar(0, 69, 255), 2);
		}
	}

	// For Orange
	vector<vector<Point>> contoursO;
	vector<Vec4i> hierarchyO;

	findContours(img_dilO, contoursO, hierarchyO, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> conPolyO(contoursO.size());
	vector<Rect> boundRectO(contoursO.size());

	// Itterate through all Orange polygons detected in the image, bound and write text for each.
	for (int i = 0; i < contoursO.size(); i++) {

		int areaO = contourArea(contoursO[i]);

		if (areaO > 1000) {

			float periO = arcLength(contoursO[i], true);
			approxPolyDP(contoursO[i], conPolyO[i], 0.02 * periO, true);
			drawContours(input_img, conPolyO, i, Scalar(255, 0, 255), 2);
			boundRectO[i] = boundingRect(conPolyO[i]);
			rectangle(input_img, boundRectO[i].tl(), boundRectO[i].br(), Scalar(0, 225, 0), 5);

			// Write sign type near bounding rect
			putText(input_img, "Road Work Sign", { boundRectO[i].x, boundRectO[i].y - 5, }, FONT_HERSHEY_PLAIN, 1, Scalar(0, 69, 255), 2);
		}
	}

	// For Green
	vector<vector<Point>> contoursG;
	vector<Vec4i> hierarchyG;

	findContours(img_dilG, contoursG, hierarchyG, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> conPolyG(contoursG.size());
	vector<Rect> boundRectG(contoursG.size());

	// Itterate through all Green polygons detected in the image, bound and write text for each.
	for (int i = 0; i < contoursG.size(); i++) {

		int areaG = contourArea(contoursG[i]);

		if (areaG > 1000) {

			float periG = arcLength(contoursG[i], true);
			approxPolyDP(contoursG[i], conPolyG[i], 0.02 * periG, true);
			drawContours(input_img, conPolyG, i, Scalar(255, 0, 255), 2);
			boundRectG[i] = boundingRect(conPolyG[i]);
			rectangle(input_img, boundRectG[i].tl(), boundRectG[i].br(), Scalar(0, 225, 0), 5);

			// Write sign type near bounding rect
			putText(input_img, "Interchange / Street Sign", { boundRectG[i].x, boundRectG[i].y - 5, }, FONT_HERSHEY_PLAIN, 1, Scalar(0, 69, 255), 2);
		}
	}

	// Write final classified img to the "classified_img" directory

	imwrite("classified_image.png", input_img);
	waitKey(0);

	return(0);
}