#ifndef VASR_LANDMARK_FLANDMARK_HPP
#define VASR_LANDMARK_FLANDMARK_HPP

#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <face-landmark/FaceLandmark.hpp>
#include <flandmark_detector.h>


namespace vasr {
namespace landmark {


class FLandmark : public FaceLandmark
{
    public:
        FLandmark( FLANDMARK_Model *model );
        virtual ~FLandmark();

        bool detect( IplImage* input, const CvRect *faceRect );
        bool getLandmark( int index, CvPoint &point );

    public:
        FLANDMARK_Model *model;
        double *landmarks;
};


}}


#endif // VASR_LANDMARK_FLANDMARK_HPP
