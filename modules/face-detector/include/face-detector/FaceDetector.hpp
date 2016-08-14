#ifndef FACE_DETECTOR_HPP
#define FACE_DETECTOR_HPP

#include <opencv2/opencv.hpp>


namespace vasr {
namespace detector {


class FaceDetector
{
    public:
        FaceDetector( );
        virtual ~FaceDetector();

        virtual bool detect( const cv::Mat &input, cv::Rect &rect ) = 0;
};


}}

#endif // FACE_DETECTOR_HPP
