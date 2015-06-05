#include <face-detector/ViolaJones.hpp>


namespace vasr {
namespace detector {


ViolaJones::ViolaJones( CvHaarClassifierCascade* cascade )
{
    this->storage = cvCreateMemStorage(0);
    cvClearMemStorage(storage);
    this->cascade = cascade;
}


ViolaJones::~ViolaJones()
{
    cvReleaseMemStorage(&storage);
}


bool ViolaJones::detect( IplImage* input, CvRect *rect )
{
    int size;
    CvSeq *rects;
    CvRect *current;
    int width = 0;

    size = std::min(input->width, input->height) / 4;
    CvSize minSize = cvSize(size, size);

    rects = cvHaarDetectObjects(input, cascade, storage, 1.2f, 2, CV_HAAR_DO_CANNY_PRUNING, minSize);
    if (rects != NULL && rects->total > 0)
    {
        // look for the bigger face
        for (int i = 0; i < rects->total; ++i)
        {
            current = (CvRect*)cvGetSeqElem(rects, i);
            if (current->width > width)
            {
                *rect = *current;
                width = current->width;
            }
        }
        return true;
    }
    return false;
}


}}
