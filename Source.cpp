#include "stdafx.h"



int main()
{
	cv::Rect roi1, roi2;
	cv::Mat img1 = cv::imread("./im0.png", 0);
	cv::Mat img2 = cv::imread("./im1.png", 0);
	cv::Mat img = cv::imread("./im0.png", 1);

	SfMFeatures matcher;
	vector<cv::DMatch> matches, outMatches; // matches for initial match method, outmatches for ransac method
	vector<cv::KeyPoint> keypoints1, keypoints2;
	vector<cv::Point2f> points1, points2;
	cv::Mat K;
	cv::Mat  K1 = (cv::Mat_<double>(3, 3) << 1625.408, 0.000, 199.345,
											0.000, 1625.408, 240.165,
											0.000, 0.000, 1.000);
	cv::Mat  K2 = (cv::Mat_<double>(3, 3) << 1625.408, 0.000, 284.364,
											0.000, 1625.408, 240.165,
											0.000, 0.000, 1.000);

	cv::Mat descriptor1, descriptor2; //descriptor vector of 2 first image
	matcher.match(img1, img2, matches, keypoints1, keypoints2, descriptor1, descriptor2);
	K = matcher.fundamentalMatrix(matches, keypoints1, keypoints2, outMatches, points1, points2);
	cv::Mat R1, R2;
	cv::Mat  P1, P2, Q, R, t;
	cv::Mat mask;
	cv::Mat E1 = cv::findEssentialMat(cv::Mat(points1), cv::Mat(points2), K1, CV_RANSAC, 0.999, 1.0, mask);
	vector<cv::Point2f> good_points1, good_points2;
	for (int i = 0; i < points2.size(); i++)
	{
		if (mask.at<uchar>(i) == 1)
		{
			good_points1.push_back(points1[i]);
			good_points2.push_back(points2[i]);

		}
	}
	mask.release();
	
	cv::Mat E = cv::findEssentialMat(cv::Mat(good_points1), cv::Mat(good_points2), K1, CV_RANSAC, 0.999, 1.0, mask); //recompute
	cv::recoverPose(E, cv::Mat(good_points1), cv::Mat(good_points2), K1, R, t, mask);
	cv::stereoRectify(K1, cv::Mat(), K2, cv::Mat(), img1.size(), R, t, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, img1.size(), &roi1, &roi2);
	
	
	/*/////////Stereo matching
	cv::Mat disparity, disparity8;
	
	 cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16, 9);
	 bm->setROI1(roi1);
	 bm->setROI2(roi2);
	 bm->setPreFilterCap(31);
	 bm->setBlockSize(13);
	 bm->setMinDisparity(-11);
	 bm->setNumDisparities(96);
	 bm->setTextureThreshold(93);
	 bm->setUniquenessRatio(15);
	 bm->setSpeckleWindowSize(100);
	 bm->setSpeckleRange(32);
	 bm->setDisp12MaxDiff(1);
	
	 bm->compute(img1, img2, disparity);

	 //-- Check its extreme values
	 double minVal; double maxVal;
	 cv::minMaxLoc(disparity, &minVal, &maxVal);
	 //-- 4. Display it as a CV_8UC1 image
	 disparity.convertTo(disparity8, CV_8UC1, 255 / (maxVal - minVal));
	 cv::imshow("disparity", disparity8);
		cv::Mat depthmap;
		cv::reprojectImageTo3D(disparity8, depthmap, Q, true, CV_32F);
		cv::imshow("depthmaps", depthmap);
		cv::waitKey(0);
		*/

	////////////Dense Optical Flow/////////////
	cv::Mat flow;
	vector<cv::Point2f> img1_points, tracked_points;
	cv::calcOpticalFlowFarneback(img1, img2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
	for (int y = 0; y < img1.rows; y += 5)
	{
		for (int x = 0; x < img1.cols; x += 5)
		{
			cv::Point2f flowPoint = flow.at<cv::Point2f>(y, x);
			tracked_points.push_back(cv::Point2f(x+flowPoint.x,y+flowPoint.y));
			img1_points.push_back(cv::Point2f(x, y));
		}
	}
	cv::Mat  P,P4,points4d,tran;
	cv::Mat zeros = (cv::Mat_<double>(3, 1) << 0, 0, 0);
	cv::hconcat(R, t, tran);
	cv::hconcat(K1, zeros, P);
	P4 = K1*tran;
	cv::triangulatePoints(P, P4, cv::Mat(img1_points), cv::Mat(tracked_points), points4d);
	cv::Mat PointCloud_m;
	cv::convertPointsFromHomogeneous(cv::Mat(points4d.t()).reshape(4, 1), PointCloud_m); //convert from homogeneous to Euclidean 
	vector<cv::Point3f> point_cloud;
	point_cloud.resize(PointCloud_m.rows);
	PointCloud_m.copyTo(cv::Mat(point_cloud));

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (int i = 0; i < point_cloud.size(); i++)
	{
		pcl::PointXYZRGB point_pcl;

		point_pcl.x = point_cloud[i].x;
		point_pcl.y = point_cloud[i].y;
		point_pcl.z = point_cloud[i].z;
		//get R,G,B value of 2d points

		cv::Vec3b color = img.at<cv::Vec3b>(cv::Point(img1_points[i].x, img1_points[i].y));
		uint8_t r = (uint8_t)color.val[2];
		uint8_t g = (uint8_t)color.val[1];
		uint8_t b = (uint8_t)color.val[0];
		uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
		point_pcl.rgb = *reinterpret_cast<float*>(&rgb);
		point_cloud_ptr->push_back(point_pcl);
	}

		pcl::visualization::CloudViewer viewer("Cloud Viewer");
	viewer.showCloud(point_cloud_ptr);
	while (!viewer.wasStopped())
	{
	}

	////////////////////////////PCL//////////////
	/*	vector<cv::Point3f> point_cloud;
		vector<uint32_t> rgb_vector;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

		for (int x = 0; x < img.rows; x++)
		{
		cv::Vec3b* row = img.ptr<cv::Vec3b>(x);

		for (int y = 0; y < img.cols; y++)
		{
		pcl::PointXYZRGB point_pcl;
		cv::Point3f p = depthmap.at<cv::Point3f>(x, y);
		if (p.z >= 10000) continue;  // Filter errors
		point_pcl.x = p.x;
		point_pcl.y = p.y;
		point_pcl.z = p.z;
		cv::Vec3b color = row[y];
		uint8_t r = (uint8_t)color.val[2];
		uint8_t g = (uint8_t)color.val[1];
		uint8_t b = (uint8_t)color.val[0];
		uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
		point_pcl.rgb = *reinterpret_cast<float*>(&rgb);
		point_cloud_ptr->push_back(point_pcl);
		}
		}

		pcl::visualization::CloudViewer viewer("Cloud Viewer");
		viewer.showCloud(point_cloud_ptr);
		while (!viewer.wasStopped())
		{
		}
		*/
		return 0;
}