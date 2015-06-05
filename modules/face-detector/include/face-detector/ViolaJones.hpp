#ifndef VIOLA_JONES_FACE_DETECTOR_HPP
#define VIOLA_JONES_FACE_DETECTOR_HPP

#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <face-detector/FaceDetector.hpp>


namespace vasr {
namespace detector {


class ViolaJones : public FaceDetector
{
    public:
        ViolaJones( CvHaarClassifierCascade* cascade );
        virtual ~ViolaJones();

        bool detect( IplImage* input, CvRect *rect );

    private:
        CvMemStorage *storage;
        CvHaarClassifierCascade *cascade;

};


}}


#endif // VIOLA_JONES_FACE_DETECTOR_HPP
