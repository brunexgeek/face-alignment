#ifndef OPENCV_EXTRAS
#define OPENCV_EXTRAS


#include <opencv2/opencv.hpp>


namespace ert {


cv::Point2f operator*(
	const cv::Mat &M,
	const cv::Point2f& p );


cv::Point2f operator+(
	const cv::Point2f& p1,
	const cv::Point2f& p2 );


cv::Point2f operator/(
	const cv::Point2f &p1,
	size_t i );


};


#endif //  OPENCV_EXTRAS
