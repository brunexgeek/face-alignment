#include <ert/opencv.hh>

namespace ert {


cv::Point2f operator*(
	const cv::Mat &M,
	const cv::Point2f& p )
{
	cv::Mat src(2, 1, CV_64F);

	src.at<double>(0,0) = p.x;
	src.at<double>(1,0) = p.y;
	//src.at<float>(2,0)=1.0;

	cv::Mat dst = M*src;
	return cv::Point2f(dst.at<double>(0,0),dst.at<double>(1,0));
}


cv::Point2f operator+(
	const cv::Point2f& p1,
	const cv::Point2f& p2 )
{
	return cv::Point2f( p1.x + p2.x, p1.y + p2.y );
}


cv::Point2f operator/(
	const cv::Point2f &p1,
	size_t i )
{
	return cv::Point2f( p1.x / i, p1.y / i );
}


}
