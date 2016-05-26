#include <opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <vector>
#include <iostream>
#include <time.h>
#include <algorithm>
#include <math.h>
#include <unordered_map>
#include <io.h>
#include <string>
#include <list>
#include <set>
#define M_PI 3.14159265358979323846


using namespace std;
using namespace cv;

//ratio must <= 0.5
void getEdgeKeypoint(int w, int h, double ratio,const vector<KeyPoint>& kp,const Mat& des, vector<KeyPoint>& kpEdge, Mat& desEdge)
{
	kpEdge.clear();
	desEdge.release();

	Mat mask = Mat::zeros(h, w, CV_8U);
	Rect roi(w*ratio, h*ratio, w*(1.0f - 2.0f*ratio), h*(1.0f - 2.0f*ratio));
	mask(roi) = 1;
	for (int i = 0; i < kp.size(); ++i)
	{
		if (mask.at<uchar>(kp[i].pt) == 0)
		{
			kpEdge.push_back(kp[i]);
			desEdge.push_back(des.row(i));
		}
	}
	return;
}

void drawPathList(const Mat& srt, Mat& dst, vector<vector<Point2f>> pathList)
{
	srt.copyTo(dst);
	for (auto path : pathList)
	{
		if (path.size() < 2)
			continue;
		Point2f ptPrev = path[0];
		for (auto pt : path)
		{
			double dx = pt.x - ptPrev.x;
			double dy = pt.y - ptPrev.y;
			double len = sqrt(dx*dx + dy*dy);
			if (len>0.05*720){
				ptPrev = pt;
				continue;
			}
			circle(dst, pt, 1, Scalar(0, 0, 255));
			line(dst, ptPrev, pt, Scalar(0, 255, 0));
			ptPrev = pt;
		}
	}
	return;
}

void calMotionHist(vector<vector<Point2f>>& pathList, vector<double>& motionHist){
	motionHist = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	for (auto path : pathList)
	{
		if (path.size() < 2)
			continue;
		Point2f last = path[0];
		for (int i = 0; i < path.size(); ++i)
		{
			Point2f current = path[i];
			double dx = current.x - last.x;
			double dy = current.y - last.y;
			double len = sqrt(dx*dx + dy*dy);
			if (len>0.05 /*|| len<0.005*/)
				continue;
			int idx = (int)floor(4 * atan2(dy, dx)/M_PI + 4) % 8;
			//cout << "dx: " << dx << "\tdy: " << dy << "\tidx: " << idx << endl;
			motionHist[idx] += len;
			last = current;
		}
	}
	double sum = 0.0;
	for (double h : motionHist)
	{
		sum += h;
	}
	for (double& h : motionHist)
	{
		h /= sum;
	}
	return;
}


//keypoint
SURF keypointDetectorAnddescriptor;
BFMatcher matcher;

int main(int argc, char* argv[])
{
	//video input
	string videoName("A_kind_of_a_Show.avi");
	VideoCapture capture(videoName);
	if (!capture.isOpened())
	{
		cout << "!capture.isOpened()";
		return -1;
	}

	//path list
	vector<vector<Point2f>> pathList;
	vector<int> kpIdx2pathListIdx;
	
	vector<KeyPoint> kpTrackedPrev;
	Mat desTrackedPrev;
	vector<KeyPoint> kpEdgePrev;
	Mat desEdgePrev;

	//firstFrame init
	Mat firstFrame;
	Mat frame, framePrev;
	capture.read(firstFrame);
	keypointDetectorAnddescriptor.detect(firstFrame, kpTrackedPrev);
	keypointDetectorAnddescriptor.compute(firstFrame, kpTrackedPrev, desTrackedPrev);
	getEdgeKeypoint(firstFrame.cols, firstFrame.rows, 0.25,
		kpTrackedPrev, desTrackedPrev,
		kpEdgePrev, desEdgePrev);
	for (int i = 0; i < kpTrackedPrev.size(); ++i)
	{
		pathList.push_back(vector<Point2f>());
		pathList[i].push_back(kpTrackedPrev[i].pt);
		kpIdx2pathListIdx.push_back(i);
	}
	firstFrame.copyTo(framePrev);

	//video writer
	VideoWriter vw("result.avi", CV_FOURCC('M', 'J', 'P', 'G'), 12, Size(firstFrame.cols, firstFrame.rows));
	if (!vw.isOpened())
		return -1;

	//frame
	vector<KeyPoint> kpCur;
	Mat desCur;
	int frameIdx = 0;

	//processing
	while (capture.read(frame))
	{
		++frameIdx;

		keypointDetectorAnddescriptor.detect(frame, kpCur);
		keypointDetectorAnddescriptor.compute(frame, kpCur, desCur);

		//edge keypoint matching for homography
		vector<Point2f> ptEdgeCurMatched;
		vector<Point2f> ptEdgePrevMatched;
		vector<vector<DMatch>> vvmatchs;
		matcher.knnMatch(desEdgePrev, desCur, vvmatchs, 2);
		for (int i = 0; i < vvmatchs.size(); ++i)
		{
			if (vvmatchs[i][0].distance < vvmatchs[i][1].distance * 0.8)
			{
				ptEdgeCurMatched.push_back(kpCur[vvmatchs[i][0].trainIdx].pt);
				ptEdgePrevMatched.push_back(kpEdgePrev[vvmatchs[i][0].queryIdx].pt);
			}
		}

		//findHomography
		Mat h = findHomography(ptEdgePrevMatched,ptEdgeCurMatched, RANSAC);
		cout << h << endl;
		
		// camera movement compensation
		for (vector<Point2f>& path : pathList){
			perspectiveTransform(path, path, h);
		}

		Mat warpedframe;
		warpPerspective(framePrev, warpedframe, h, frame.size());
		imshow("frame", frame);
		imshow("prev", framePrev);
		imshow("warpedframe", warpedframe);

		getEdgeKeypoint(frame.cols, frame.rows, 0.25,
			kpCur, desCur,
			kpEdgePrev, desEdgePrev);
		frame.copyTo(framePrev);

		//keypoint tracking for pathlist
		vector<int> kpIdx2pathListIdxTemp;
		vector<KeyPoint> kpTrackedCur;
		Mat desTrackedCur;
		set<int> curMatchedKpIdxSet;
		matcher.knnMatch(desTrackedPrev, desCur, vvmatchs, 2);
		for (int i = 0; i < vvmatchs.size(); ++i)
		{
			if (vvmatchs[i][0].distance < vvmatchs[i][1].distance * 0.6)
			{
				pathList[kpIdx2pathListIdx[i]].push_back(kpCur[vvmatchs[i][0].trainIdx].pt);
				kpTrackedCur.push_back(kpCur[vvmatchs[i][0].trainIdx]);
				desTrackedCur.push_back(desCur.row(vvmatchs[i][0].trainIdx));
				kpIdx2pathListIdxTemp.push_back(kpIdx2pathListIdx[i]);
				curMatchedKpIdxSet.insert(vvmatchs[i][0].trainIdx);
			}
		}
		if (frameIdx%5==0)
		{
			//add new keypoint
			for (int i = 0; i < kpCur.size(); ++i)
			{
				if (curMatchedKpIdxSet.find(i) == curMatchedKpIdxSet.end()){
					kpTrackedCur.push_back(kpCur[i]);
					desTrackedCur.push_back(desCur.row(i));
					pathList.push_back(vector<Point2f>());
					pathList.rbegin()->push_back(kpCur[i].pt);
					kpIdx2pathListIdxTemp.push_back(pathList.size() - 1);
				}
			}
		}

		kpIdx2pathListIdx = kpIdx2pathListIdxTemp;
		kpTrackedPrev = kpTrackedCur;
		desTrackedCur.copyTo(desTrackedPrev);
		Mat show;
		drawPathList(frame, show, pathList);
		imshow("pathlist", show);
		waitKey(1);

		vw << show;
	}
	vw.release();
	//uniform
	for (vector<Point2f>& path : pathList)
	{
		for (Point2f& pt : path)
		{
			pt.x /= firstFrame.cols;
			pt.y /= firstFrame.rows;
		}
	}

	vector<double> motionHist;
	calMotionHist(pathList, motionHist);
	cout << "h : " << endl;
	for (double h : motionHist)
		cout << h << " ";
	cout << endl;
	return 1;
}