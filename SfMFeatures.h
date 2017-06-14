#ifndef FEATURES2D
#define FEATURES2D

#include "stdafx.h"


using namespace std;
struct pointcloud;

	class SfMFeatures {
	public:
		SfMFeatures():ratio (0.8f) {
			//SIFT 
			detector = cv::xfeatures2d::SIFT::create();
			extractor = cv::xfeatures2d::SIFT::create();
			//BruteMatcher with Euclidean distance 
			matcher = cv::makePtr<cv::BFMatcher>((int)cv::NORM_L2, false);
		}
		virtual ~SfMFeatures();
		//Set the feature detector
		 void setFeatureDetector(const cv::Ptr<cv::FeatureDetector>& detect) { detector = detect; }
		//Set the descriptor extractor
		 void setExtractorDescriptor(const cv::Ptr <cv::DescriptorExtractor>& desc) { extractor = desc; }
		//Set the matcher
		 void setMatcher(const cv::Ptr <cv::DescriptorMatcher>& match) { matcher = match; }
		 //DetectFeature
		 void detectFeature(const cv::Mat& img, vector<cv::KeyPoint>& keypoints);
		//Detect features using Dense SIFT
		 void DenseSIFT(const cv::Mat& img, vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
		 //Compute descriptor vector of an image
		 void computeDescriptor(const cv::Mat& img, vector<cv::KeyPoint>& keypoints, cv::Mat& descriptor);
		//Set ratio parameter for ratio test
		 void setRatio(float rat) { ratio = rat; }
		//Clear matches for which KNN ratio > threshold, return removed points
		 	int ratioTest(vector<vector<cv::DMatch>>& matches);
		//Symmetrical matches
		  void symmetryTest(const vector<vector<cv::DMatch>>& matches1, const vector<vector<cv::DMatch>>& matches2,
			vector<cv::DMatch>& symMatches);
		//Matching for twoview geometry
		 	 void match(cv::Mat& img1, cv::Mat& img2, vector<cv::DMatch>& matches,
			vector<cv::KeyPoint>& keypoints1, vector<cv::KeyPoint>& keypoints2, cv::Mat& descriptor1, cv::Mat& descriptor2);
		//Rematching for multiview geometry
		 void rematch(const cv::Mat& frame,vector<cv::DMatch>& good_matches,vector<cv::KeyPoint>& keypoints_frame,
			const cv::Mat& descriptors_model, cv::Mat& descriptors_frame);
		//Compute fundamental matrix and remove outliers
		cv::Mat fundamentalMatrix( const vector<cv::DMatch>& matches, vector<cv::KeyPoint>& keypoints1,
			vector<cv::KeyPoint>& keypoints2, vector<cv::DMatch> &outMatches, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2);
	private:
		//pointer to the feature point detector object
		 cv::Ptr<cv::FeatureDetector> detector;
		//pointer to the feature descriptor extractor object
		 	cv::Ptr <cv::DescriptorExtractor> extractor;
		//pointer to the matcher object
		 cv::Ptr <cv::DescriptorMatcher> matcher;
		//ratio K-nearest neighbor algorithm
			float ratio;
	};




#endif 