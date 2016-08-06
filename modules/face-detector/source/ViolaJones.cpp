#include <face-detector/ViolaJones.hpp>


namespace vasr {
namespace detector {


using namespace cv;


ViolaJones::ViolaJones( CascadeClassifier *cascade )
{
    /*this->storage = MemoryStorage(0);
    cvClearMemStorage(storage);*/
    this->cascade = cascade;
}


ViolaJones::~ViolaJones()
{
    //cvReleaseMemStorage(&storage);
}


bool ViolaJones::detect( const Mat &input, Rect &rect )
{
    int size;
    Rect current;
    int width = 0;
   std::vector<Rect> objects;

    size = std::min(input.cols, input.rows) / 4;
    Size minSize = Size(size, size);

    cascade->detectMultiScale(input, objects, 1.2f, 2, 0, minSize);
    if (!objects.empty())
    {
        // look for the bigger face
        for (size_t i = 0; i < objects.size(); ++i)
        {
            current = objects.at(i);
            if (current.width > width)
            {
                rect = current;
                width = current.width;
            }
        }
        return true;
    }
    return false;
}


}}
