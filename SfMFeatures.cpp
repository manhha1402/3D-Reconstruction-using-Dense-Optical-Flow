#include "SfMFeatures.h"

using namespace std;
 
	SfMFeatures::~SfMFeatures()
	{
		//TODO 
	}
	void SfMFeatures::detectFeature(const cv::Mat& img, vector<cv::KeyPoint>& keypoints)
	{
		detector->detect(img, keypoints); //return keypoints(features) of image to keypoints
	}
	void SfMFeatures::DenseSIFT(const cv::Mat& img, vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
	{
		cv::Ptr<cv::xfeatures2d::SIFT> denseSIFT = cv::xfeatures2d::SIFT::create();
		
		int step = 10;
		for (int x = step; x < img.rows - step; x += step)
		{
			for (int y = step; y < img.cols - step; y += step)
			{
				keypoints.push_back(cv::KeyPoint(float(y), float(x), float(step)));
			}
		}
		denseSIFT->compute(img, keypoints, descriptors);

	}

	 void SfMFeatures::computeDescriptor(const cv::Mat& img, vector<cv::KeyPoint>& keypoints, cv::Mat& descriptor)
	{
		extractor->compute(img, keypoints, descriptor); //return description vector of image to descriptor
	}
	int SfMFeatures::ratioTest(vector<vector<cv::DMatch>>& matches)
	{
		int removed = 0;
		for (vector<vector<cv::DMatch>>::iterator i = matches.begin(); i != matches.end(); i++)
		{
			if (i->size() > 1)
			{
				//Check KNN
				if ((*i)[0].distance > ratio*(*i)[1].distance)
				{
					i->clear();
					removed++;
				}
			}
			else //does not have 2 neighbors
			{
				i->clear();
				removed++;
			}
		}
		return removed;
	}
	void SfMFeatures::symmetryTest(const vector<vector<cv::DMatch>>& matches1, const vector<vector<cv::DMatch>>& matches2,
		vector<cv::DMatch>& symMatches)
	{
		//For all matches from img1 -> img2
		for (vector<vector<cv::DMatch>>::const_iterator i1 = matches1.begin(); i1 != matches1.end(); i1++)
		{
			//ignore deleted matches and does not have 2 neighbors 
			if (i1->empty() || i1->size() < 2) continue;
			//For all matches from img2->img1
			for (vector<vector<cv::DMatch>>::const_iterator i2 = matches2.begin(); i2 != matches2.end(); i2++)
			{
				//ignore deleted matches and does not have 2 neighbors 
				if (i2->empty() || i2->size() < 2) continue;
				//Match symmetry test
				if ((*i1)[0].queryIdx == (*i2)[0].trainIdx && (*i2)[0].queryIdx == (*i1)[0].trainIdx)
				{
					symMatches.push_back(cv::DMatch((*i1)[0].queryIdx, (*i1)[0].trainIdx, (*i1)[0].distance));
					break;
				}

			}

		}
	}
	void SfMFeatures::match(cv::Mat& img1, cv::Mat& img2, vector<cv::DMatch>& matches,
		vector<cv::KeyPoint>& keypoints1, vector<cv::KeyPoint>& keypoints2, cv::Mat& descriptor1, cv::Mat& descriptor2)
	{	//1a.Detect features of 2 images
		
		detector->detect(img1, keypoints1);
		detector->detect(img2, keypoints2);

		//1b. Extract SIFT descriptors
		cv::Mat  img1_keypoint, img2_keypoint,imgs_keypoints;
		extractor->compute(img1, keypoints1, descriptor1);
		extractor->compute(img2, keypoints2, descriptor2);
		//Draw image with keypoints
		cv::drawKeypoints(img1, keypoints1, img1_keypoint);
		cv::drawKeypoints(img2, keypoints2, img2_keypoint);
		cv::hconcat(img1_keypoint, img2_keypoint, imgs_keypoints);
		cv::imwrite("./keypoints.jpg", imgs_keypoints);
		
		
		//2. Match 2 image descriptors
		vector<vector<cv::DMatch>> matches1, matches2;
		//from img1-> img2
		matcher->knnMatch(descriptor1, descriptor2, matches1, 2);
		//from img2->img1
		matcher->knnMatch(descriptor2, descriptor1, matches2, 2);
		//3.Remove matches for which KNN> threshold
		//Clean img1->img2
		ratioTest(matches1);
		//clean img2->img1
		ratioTest(matches2);
		//4. Symmetry Test
		symmetryTest(matches1, matches2, matches);
	}
	void SfMFeatures::rematch(const cv::Mat& frame, vector<cv::DMatch>& good_matches, vector<cv::KeyPoint>& keypoints_frame,
		const cv::Mat& descriptors_model, cv::Mat& descriptors_frame)
	{
		// 1a. Detection of the SIFT features
		this->detectFeature(frame,keypoints_frame);
		// 1b. Extraction of the SIFT descriptors of frame
		this->computeDescriptor(frame, keypoints_frame,descriptors_frame);
		// 2. Match the two image descriptors
		vector<vector<cv::DMatch> > matches12, matches21;	
		// 2a. From image 1 to image 2
		matcher->knnMatch(descriptors_frame, descriptors_model, matches12, 2); // return 2 nearest neighbours
		// 2b. From image 2 to image 1
		matcher->knnMatch(descriptors_model, descriptors_frame, matches21, 2); // return 2 nearest neighbours
		// 3. Remove matches for which NN ratio is > than threshold
		// clean image 1 -> image 2 matches
		ratioTest(matches12);
		// clean image 2 -> image 1 matches
		ratioTest(matches21);

		// 4. Remove non-symmetrical matches
		symmetryTest(matches12, matches21, good_matches);
	}
	cv::Mat SfMFeatures::fundamentalMatrix(const vector<cv::DMatch>& matches, vector<cv::KeyPoint>& keypoints1,
		vector<cv::KeyPoint>& keypoints2, vector<cv::DMatch> &outMatches, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2)
	{
		//Convert keypoints to coordinates
		for (vector<cv::DMatch>::const_iterator i = matches.begin(); i != matches.end(); i++)
		{
			points1.push_back(cv::Point2f(keypoints1[i->queryIdx].pt.x, keypoints1[i->queryIdx].pt.y));
			points2.push_back(cv::Point2f(keypoints2[i->trainIdx].pt.x, keypoints2[i->trainIdx].pt.y));
		}
		//Compute F using Ransac
		vector<uchar> inliers(points1.size(), 0);
		cv::Mat fundamental = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2), inliers, CV_FM_RANSAC, 3.0, 0.99);
		//Extract inlier matches inside margin
		vector<uchar>::const_iterator itIn = inliers.begin();
		vector<cv::DMatch>::const_iterator itM = matches.begin();
		for (; itIn != inliers.end(); ++itIn, ++itM)
		{
			if (*itIn)
			{
				//match inside margin
				outMatches.push_back(*itM);
			}
		}
		//Fundamental matrix is recomputed with outMatches
		points1.clear(); points2.clear();
		//Convert keypoints to coordinates with outMatches	
		for (vector<cv::DMatch>::const_iterator i = outMatches.begin(); i != outMatches.end(); i++)
		{
			points1.push_back(cv::Point2f(keypoints1[i->queryIdx].pt.x, keypoints1[i->queryIdx].pt.y));
			points2.push_back(cv::Point2f(keypoints2[i->trainIdx].pt.x, keypoints2[i->trainIdx].pt.y));

		}
		
		//Compute F using 8 correspondences
		fundamental = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2), CV_FM_8POINT);
		return fundamental;

	}
