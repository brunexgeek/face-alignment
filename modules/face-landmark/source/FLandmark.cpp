#include <face-landmark/FLandmark.hpp>


namespace vasr {
namespace landmark {


FLandmark::FLandmark( FLANDMARK_Model *model )
{
    this->model = model;
    landmarks = new double[2 * model->data.options.M];//(double*) malloc(2 * model->data.options.M * sizeof(double));
}


FLandmark::~FLandmark()
{
    delete[] landmarks;
}


bool FLandmark::detect( IplImage* input, const CvRect *faceRect )
{
    //int bwmargin[2] = { 5, 5 };

    memset(landmarks, 0, 2 * model->data.options.M * sizeof(double));

    if (faceRect == NULL) return false;

    int bbox[4];
    bbox[0] = faceRect->x;
    bbox[1] = faceRect->y;
    bbox[2] = faceRect->x + faceRect->width;
    bbox[3] = faceRect->y + faceRect->height;

    return flandmark_detect(input, bbox, model, landmarks/*, bwmargin*/) == 0;
}


bool FLandmark::getLandmark( int index, CvPoint &point )
{
    if (index < 0 || index >= model->data.options.M) return false;

    point.x = landmarks[index * 2];
    point.y = landmarks[index * 2 + 1];

    return true;
}


}}
