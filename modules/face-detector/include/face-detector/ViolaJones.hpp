#ifndef VIOLA_JONES_FACE_DETECTOR_HPP
#define VIOLA_JONES_FACE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <face-detector/FaceDetector.hpp>


namespace vasr {
namespace detector {


class ViolaJones : public FaceDetector
{
    public:
        ViolaJones( cv::CascadeClassifier *cascade );
        virtual ~ViolaJones();

        bool detect( const cv::Mat &input, cv::Rect &rect );

    private:
        cv::MemStorage *storage;
        cv::CascadeClassifier *cascade;

};


}}


#endif // VIOLA_JONES_FACE_DETECTOR_HPP
