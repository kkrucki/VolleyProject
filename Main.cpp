#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <opencv2/tracking/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>


using namespace cv;
using namespace std;


struct ball
{
	Point center = Point(-1,-1);
	Rect bbox = Rect(0,0,0,0);
};

int determineBallRegion(Mat img, Point center);
struct ball findBall(Mat img);
Point calcImageShift(Mat img, Mat refFrame);
int determineBoardRegions(Mat img);

vector<vector<Point>> boardRegions;
vector<int> regionIdx(4, 0);

int main(int argc, char *argv[])
{
	bool debug = false;
	
	//Open Video
	VideoCapture video;
	string vidName = argv[1];
	cout << vidName;
	video.open(vidName);
	if (!video.isOpened())
	{
		cout << "Could not open video";
		return -1;
	
	}

	//Setup Output Video
	int h = video.get(CAP_PROP_FRAME_HEIGHT);
	int w = video.get(CAP_PROP_FRAME_WIDTH);
	int fps = video.get(CAP_PROP_FPS);

	VideoWriter vw("Result.mp4", VideoWriter::fourcc('M', 'P', '4', 'V'), fps, Size(w, h), true);

	//Read Initial Reference Frame
	Mat frame, refFrame;
	video.read(refFrame);

	//Determine initial board regions and store in regionIdx
	determineBoardRegions(refFrame);

	while (video.read(frame))
	{
		//Find ball center and bounding box in each frame
		struct ball bb = findBall(frame);

		//If ball is found
		if (bb.center.x != -1)
		{
			//Calculate image shift from initial reference frame
			Point shift = calcImageShift(frame, refFrame);
			//Determine region ball is in
			int reg = determineBallRegion(frame, bb.center - shift);

			//Draw ball center, bounding box, and indicate region
			circle(frame, bb.center, 2, Scalar(0, 0, 255), 5);
			//circle(refFrame, bb.center-shift, 2, Scalar(0, 0, 255), 10);

			rectangle(frame, bb.bbox, Scalar(0, 255, 0), 3);
			string txt;
			if(reg == 0)
				txt = "BALL DETECTED ON LINE";
			else
				txt = "BALL DETECTED IN REGION " + to_string(reg);

			putText(frame, txt, Point(20,20), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,0),2);
		}
		else
		{
			putText(frame, "NO BALL DETECTED", Point(20, 20), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 0),2);

		}
		
		vw.write(frame);
		if (debug)
		{
			imshow("img", frame);
			waitKey(1);
		}
	}

	vw.release();
}

int determineBoardRegions(Mat img)
{
	//convert to gray
	Mat grayFrame;
	cvtColor(img, grayFrame, COLOR_BGR2GRAY);

	//threshold and find contours of regions
	Mat threshImg;
	threshold(grayFrame, threshImg, 200, 255, THRESH_BINARY_INV);
	findContours(threshImg, boardRegions, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	//Find centroids of regions
	vector<int> regionXs, regionYs;
	for (int i = 0; i < boardRegions.size(); i++)
	{
		Moments m = moments(boardRegions[i]);
		regionYs.push_back(m.m01 / m.m00);
		regionXs.push_back(m.m10 / m.m00);
	}

	//Determine which region corresponds to 1-4
	double mx = 0, mn = 0; int mxIdx[2], mnIdx[2];
	minMaxIdx(regionXs, &mn, &mx, mnIdx, mxIdx);
	regionIdx[0] = mnIdx[1] + 1;
	regionIdx[2] = mxIdx[1] + 1;

	minMaxIdx(regionYs, &mn, &mx, mnIdx, mxIdx);
	regionIdx[1] = mnIdx[1] + 1;
	regionIdx[3] = mxIdx[1] + 1;


	return 0;
}

Point calcImageShift( Mat img, Mat refFrame)
{
	//Convert to gray
	Mat grayImg, grayRef;
	cvtColor(img, grayImg, COLOR_BGR2GRAY);
	cvtColor(refFrame, grayRef, COLOR_BGR2GRAY);

	//Compute cross correlation and find max point between frames
	Mat_<float> result;
	Mat normRes;
	filter2D(grayImg, result, CV_32F, grayRef);
	double mn, mx; Point mnIdx, mxIdx;
	minMaxLoc(result, &mn, &mx, &mnIdx, &mxIdx);

	//Compare to middle of refFrame to get shift
	Point shift;
	shift.x = mxIdx.x-refFrame.cols / 2;
	shift.y = mxIdx.y -refFrame.rows / 2;

	return shift;

}




struct ball findBall(Mat img)
{
	//Ball pixels will be easier to find in HSV space so convert
	Mat hsvImg;
	cvtColor(img, hsvImg, COLOR_RGB2HSV);

	//Find yellowish pixels
	Mat yellowImg;
	inRange(hsvImg, Scalar(45, 100, 100), Scalar(85, 255, 255), yellowImg);

	//Calculate center and bbox of ball
	vector<vector<Point>> ball;
	findContours(yellowImg, ball, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	Moments m;
	struct ball bb;

	if (ball.size() > 0)
	{
		//Find largest contour
		int idx = 0, sz = ball[0].size();
		for (int i = 1; i < ball.size(); i++)
		{
			if (ball[i].size() > sz)
			{
				idx = i;
				sz = ball[i].size();
			}
		}
		m = moments(ball[idx]);
		bb.center = Point(m.m10 / m.m00, m.m01 / m.m00);
		bb.bbox = boundingRect(ball[idx]);
	}
	else
		bb.center = Point(-1, -1);



	return bb;
}

int determineBallRegion(Mat img, Point center)
{
	//check each board region for ball
	for (int i = 0; i < boardRegions.size(); i++)
	{
		int in = pointPolygonTest(boardRegions[i], center, false);
		if (in == 1)
		{
			return regionIdx[i];
		}
	}
	return 0;
}

//Point findBallCenter(Mat img, Mat refFrame)
//{
//	Mat subImg;
//	absdiff( refFrame, img, subImg);
//	Mat threshImg, normSub;
//	threshold(subImg, threshImg, 50, 255, THRESH_BINARY);
//
//	int erodekernelsize = 9; //Erosion kernel size for filtering image
//	int erodekernelshape = MORPH_ELLIPSE;
//	Mat ERODE_KERNEL = getStructuringElement(erodekernelshape, cv::Size(erodekernelsize, erodekernelsize));
//	Mat eImage, dImage;
//	erode(threshImg, eImage, ERODE_KERNEL);
//	dilate(eImage, dImage, ERODE_KERNEL);
//	cv::normalize(dImage, normSub, 0, 255, cv::NORM_MINMAX, CV_8UC1);
//
//	//cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
//	//cv::normalize(refFrame, refFrame, 0, 255, cv::NORM_MINMAX, CV_8UC1);
//
//	imshow("sub", normSub);
//	//imshow("r", threshImg);
//
//	//imshow("t", img);
//
//	waitKey(1);
//	return Point(0, 0);
//}

//Mat findYellowPixels(Mat img)
//{
//	Mat hsvImg;
//	cvtColor(img, hsvImg, COLOR_RGB2HSV);
//	
//	Mat yellowImg;
//	inRange(hsvImg, Scalar(55, 100, 100), Scalar(85, 255, 255), yellowImg);
//
//	cv::normalize(yellowImg, yellowImg, 0, 255, cv::NORM_MINMAX, CV_8UC1);
//
//	imshow("yellow", yellowImg);
//	waitKey(1);
//
//	return yellowImg;
//
//}