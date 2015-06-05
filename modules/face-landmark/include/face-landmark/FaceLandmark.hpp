#ifndef VASR_LANDMARK_FACELANDMARK_HPP
#define VASR_LANDMARK_FACELANDMARK_HPP

#include <cv.h>
#include <cvaux.h>
#include <highgui.h>


namespace vasr {
namespace landmark {


class FaceLandmark
{
    public:
        FaceLandmark( );
        virtual ~FaceLandmark();

        virtual bool detect( IplImage* input, const CvRect *faceRect ) = 0;
        //virtual bool getLandmark( int index, CvPoint &point ) = 0;

        virtual int getX( int index );
        virtual int getY( int index );
};


}}


#endif // VASR_LANDMARK_FACELANDMARK_HPP
