#ifndef FACE_DETECTOR_HPP
#define FACE_DETECTOR_HPP

#include <cv.h>
#include <cvaux.h>
#include <highgui.h>


namespace vasr {
namespace detector {


class FaceDetector
{
    public:
        FaceDetector( );
        virtual ~FaceDetector();

        virtual bool detect( IplImage* input, CvRect *rect ) = 0;
};


}}

#endif // FACE_DETECTOR_HPP
