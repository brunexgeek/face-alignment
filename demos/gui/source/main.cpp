#include <cstring>
#include <stdlib.h>
#include <cv.h>
#include <cvaux.h>
#include <opencv2/opencv.hpp>
#include <getopt.h>
#include <istream>
#include <face-detector/detector.hpp>
#include <face-landmark/landmark.hpp>
#include "../../../modules/face-landmark/source/ert/ShapePredictor.hh"


using namespace vasr::detector;
using namespace vasr::landmark;
using namespace ert;
using namespace cv;


#define INFOBOX_HEIGHT        100

#define FRAME_MAX_WIDTH       640

#define FACE_CASCADE_FILE     "haarcascade_frontalface_alt2.xml"

#define FLANDMARK_FILE        "flandmark_model.dat"


const uint16_t LAYOUT_68_PARTS[] =
{
	// contorno da face
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, ObjectDetection::OPEN,
	// sobrancelha esquerda
	17, 18, 19, 20, 21, ObjectDetection::OPEN,
	// sobrancelha direita
	22, 23, 24, 25, 26, ObjectDetection::OPEN,
	// linha vertical do nariz
	27, 28, 29, 30, ObjectDetection::OPEN,
	// linha horizontal do nariz
	31, 32, 33, 34, 35, ObjectDetection::OPEN,
	// olho esquerdo
	36, 37, 38, 39, 40, 41, ObjectDetection::CLOSE,
	// olho direito
	42, 43, 44, 45, 46, 47, ObjectDetection::CLOSE,
	// parte externa da boca
	48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, ObjectDetection::CLOSE,
	// parte interna da boca
	60, 61, 62, 63, 64, 65, 66, 67, ObjectDetection::CLOSE,
	ObjectDetection::END
};


bool useCamera = false;
bool showProcessedFrame = false;
char *inputFileName = NULL;
char *outputFileName = NULL;
bool useRotation = false;

CvPoint lastLeft/* = { 0, 0 }*/;
CvPoint lastRight/* = { 0, 0 }*/;

void main_usage()
{
    fprintf(stderr, "Usage: demo_gui -i <input video file> [ -o <output video file> -p -r ]\n");
    fprintf(stderr, "       demo_gui -c [ -o <output video file> -p -r ] \n");
    exit(EXIT_FAILURE);
}


void main_parseOptions( int argc, char **argv )
{
    int opt;

    while ((opt = getopt(argc, argv, "i:o:cpr")) != -1)
    {
        switch (opt)
        {
            case 'i':
                inputFileName = optarg;
                break;
            case 'r':
                useRotation = true;
            case 'o':
                outputFileName = optarg;
                break;
            case 'c':
                useCamera = true;
                break;
            case 'p':
                showProcessedFrame = true;
                break;
            default: /* '?' */
                main_usage();
        }
    }
    if (inputFileName == NULL && useCamera == false) main_usage();
}


static int main_initDevice(
    VideoCapture **device,
    const char *fileName = NULL,
    int camera = 0,
    int width = 640,
    int height = 480)
{
    if (*device == NULL)
    {
        if (fileName == NULL)
        {
            printf("Capturing from camera #%d\n", camera);
            *device = new VideoCapture(camera);
            (*device)->set(CV_CAP_PROP_FRAME_WIDTH, width);
            (*device)->set(CV_CAP_PROP_FRAME_HEIGHT, height);
        }
        else
        {
            printf("Capturing from file '%s'\n", fileName);
            *device = new VideoCapture(fileName);
        }

        if (*device == NULL)
        {
            printf("Error connecting to capture device\n");
            return 1;
        }
    }
    return 0;
}


static void main_println( Mat &image, int x, int y, int line, const char *text )
{
    Size size;
    int baseline;

    size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    putText(image,
		text,
		Point(x, y + (line + 1) * (size.height + 10)),
		FONT_HERSHEY_SIMPLEX,
		0.5,
		Scalar(255, 255, 255, 0),
		1,
		CV_AA);
}


static void main_getDelta(
    CvPoint &left,
    CvPoint &right,
    double &dx,
    double &dy,
    CvPoint *center = NULL )
{
    // compute the center of mouth
    if (center != NULL)
    {
        center->x = left.x + (right.x - left.x) / 2;
        if (left.y < right.y)
            center->y = left.y + (right.y - left.y) / 2;
        else
            center->y = left.y - (right.y - left.y) / 2;
    }
    //dx += std::max(left.x, right.x) - std::min(left.x, right.x);
    //dy += std::max(left.y, right.y) - std::min(left.y, right.y);
    dx += (left.x) - (right.x);
    dy += (left.y) - (right.y);
}


static Mat main_getRotationMatrix(
    FaceLandmark &landmarker  )
{
    double dx = 0, dy = 0;
    CvPoint center, left, right;
    cv::Mat rot;

#if (0)
    left.x = (landmarker.getX(5) + landmarker.getX(1)) / 2;
    left.y = (landmarker.getY(5) + landmarker.getY(1)) / 2;
    right.x = (landmarker.getX(2) + landmarker.getX(6)) / 2;
    right.y = (landmarker.getY(2) + landmarker.getY(6)) / 2;
#else
    left.x = landmarker.getX(5);
    left.y = landmarker.getY(5);
    right.x = landmarker.getX(6);
    right.y = landmarker.getY(6);
#endif

    if (lastLeft.x == 0)
    {
        lastLeft.x = left.x;
        lastLeft.y = left.y;
        lastRight.x = right.x;
        lastRight.y = right.y;
    }
    else
    {
        lastLeft.x = (lastLeft.x + left.x) / 2;
        lastLeft.y = (lastLeft.y + left.y) / 2;
        lastRight.x = (right.x + left.x) / 2;
        lastRight.y = (right.y + left.y) / 2;
    }

    main_getDelta(lastLeft, lastRight, dx, dy, &center);

    //printf("delta.x = %f  delta.y = %f\n", dx, dy);
    if (dx < 0) dx *= -1;

    double m = dy / dx;
    double angle = (m == 0) ? 0 : atan(m) * -180 / 3.14159265;
//printf("Angle: %f degress gradient %f)\n", angle, m);
    return cv::getRotationMatrix2D(center, angle, 1);
}



static void main_rotateFace(
    Mat &image,
    Mat &rotated,
    FaceLandmark &landmarker )
{
    cv::Mat rot;
    cv::Point p, lp, rp;

    rot = main_getRotationMatrix(landmarker);

    p = cv::Point(landmarker.getX(3), landmarker.getY(3));
    lp = rot * p;
    p = cv::Point(landmarker.getX(4), landmarker.getY(4));
    rp = rot * p;

    float width = rp.x - lp.x;
    lp.x += -1 * width * 0.2f;
    rp.x += width * 0.2f;

    width = rp.x - lp.x;
    float height = width * 0.5f;
    lp.y += -1 * height / 3;
    rp.y += 2 * (height / 3);

    cv::Mat src, dst;
    src = image;
    warpAffine(src, dst, rot, dst.size());
    rotated = dst;

    rectangle(rotated, lp, rp, CV_RGB(255,0,0) );
}


int main( int argc, char** argv )
{
    double startTime, processTime, finalTime, frameTime;
    int result;
    bool mustResize = false;
    bool hasFace;
    ViolaJones *detector;
    const char *windowName = "Visual ASR Demo";
    VideoWriter *writer = NULL;
    int inputFPS = 0;
    int frameW = 0, originalW;
    int frameH = 0, originalH;
    int frameCount = 0;
    VideoCapture *device = NULL;
    Mat frame, tempFrame;
    char text[256];

    main_parseOptions(argc, argv);

    // initialize the input device
    if (useCamera)
        result = main_initDevice(&device, NULL, 0, 640, 480);
    else
        result = main_initDevice(&device, inputFileName);
    if (result != 0) return 1;
    originalH = frameH = (int)device->get(CV_CAP_PROP_FRAME_HEIGHT);
    originalW = frameW = (int)device->get(CV_CAP_PROP_FRAME_WIDTH);
    frameCount = (int)device->get(CV_CAP_PROP_FRAME_COUNT);
    inputFPS = (int)device->get(CV_CAP_PROP_FPS);
    if (frameCount < 0) frameCount = 0;
    if (inputFPS < 0) inputFPS = 24;
    frameTime = 1000 / inputFPS;

    // limit the width of the input frame
    if (frameW > FRAME_MAX_WIDTH)
    {
        frameH = (frameH * FRAME_MAX_WIDTH / frameW);
        frameW = FRAME_MAX_WIDTH;
        mustResize = true;
    }
    // create the display window
    namedWindow(windowName, 0);
    int windowW = frameW;
    int windowH = frameH + INFOBOX_HEIGHT;
    resizeWindow(windowName, windowW, windowH);

    // initialize the output video writer
    if (outputFileName != NULL)
        writer = new VideoWriter(outputFileName,
            CV_FOURCC('M', 'J', 'P', 'G'), inputFPS,
            cvSize(windowW, windowH));

    // load the HaarCascade classifier for face detection
    CascadeClassifier faceCascade(FACE_CASCADE_FILE);
    //faceCascade = (CvHaarClassifierCascade*)cvLoad(FACE_CASCADE_FILE, 0, 0, 0);
    if (faceCascade.empty())
    {
        printf("Error loading face cascade file '%s'\n", FACE_CASCADE_FILE);
        return 1;
    }

    // load flandmark model
    ShapePredictor model;
	std::ifstream input("model.dat");
	model.deserialize(input);
	input.close();

    // create the temporary buffers
    Rect bbox;
    Mat frame_bw = Mat::zeros(frameW, frameH, CV_8UC1);
    Mat rotated = Mat::zeros(frameW, frameH, CV_8UC1);
    Mat display = Mat::zeros(windowW, windowH, CV_8UC3);
std::cout << "'display'  Type " << display.type() << std::endl;
    int currentFrame = 0;
    bool flag = true;
    int landmarkTime = 0;

    // initialize the face detector and face landmarker
    detector = new ViolaJones(&faceCascade);

    while (flag)
    {
        if (useCamera == false && ++currentFrame >= frameCount - 2) break;

        startTime = (double)getTickCount();

        //frame = main_getFrame(&device, infname);
        *device >> frame;

        // check if need to resize the frame
        if (mustResize)
        {
            resize(frame, tempFrame, Size(frameW, frameH), 0, 0, cv::INTER_AREA);
            frame = tempFrame;
        }
        // check if need to flip horizontaly (when using camera)
        if (useCamera)
        {
            flip(frame, tempFrame, 1);
            frame = tempFrame;
		}
		//tempFrame = Mat::zeros(frame.rows, frame.cols, frame.type());
		//bilateralFilter(frame, tempFrame, 16, 32, 8);
		//frame = tempFrame;
        // convert the original frame to grayscale and look for landmarks
        cvtColor(frame, frame_bw, CV_RGB2GRAY);
        bool hasFace = detector->detect(frame_bw, bbox);
        if (showProcessedFrame)
        {
			cvtColor(frame_bw, frame, CV_GRAY2RGB);
		}
        /*if (hasFace && useRotation)
        {
            main_rotateFace(frame, rotated, *landmarker);
            frame = rotated;
        }*/

        // update the display image
        copyMakeBorder(frame, display, 0, INFOBOX_HEIGHT, 0, 0, cv::BORDER_CONSTANT, cvScalar(0.0f, 0.0f, 0.0f, 0.0f));

        // display landmarks
        if (hasFace)
        {
			ObjectDetection det = model.detect(frame_bw, bbox);

            rectangle(display, Point(bbox.x, bbox.y), cvPoint(bbox.x+bbox.width, bbox.y+bbox.height), CV_RGB(255,0,0) );
            //rectangle(display, Point(model->bb[0], model->bb[1]), Point(model->bb[2], model->bb[3]), CV_RGB(0,0,255) );
            det.plot(display, LAYOUT_68_PARTS, Scalar(255,0,0));
            /*if (!useRotation)
            {
                Point point;
                for (int i = 1; i < model->data.options.M; ++i)
                {
                    point.x = landmarker->getX(i);
                    point.y = landmarker->getY(i);
                    circle(display, point, 3, CV_RGB(255,0,0), CV_FILLED);
                }
            }*/
        }

        // update time counters
        processTime = ((double)getTickCount() - startTime) / ((double)getTickFrequency() * 1000);
        finalTime = processTime + ((frameTime > processTime) ? frameTime - processTime : 0);

        // print FPS
        sprintf(text, "FPS: Original=%d    Displayed=%.2f    Processed=%.2f",
            inputFPS,
            1000.0 / finalTime,
            1000.0 / processTime);
        main_println(display, 10, frameH, 0, text);
        // print frame count
        sprintf(text, "Frame %d of %d", currentFrame, frameCount);
        main_println(display, 10, frameH, 1, text);
        // print frame size
        sprintf(text, "Frame: Original=%dx%d    Processed=%dx%d",
            originalW,
            originalH,
            frameW,
            frameH);
        main_println(display, 10, frameH, 2, text);
        // print face size
        sprintf(text, "Face: Minimum=%dx%d    Current=%dx%d    Time=%d ms",
            std::min(frameW, frameH) / 4,
            std::min(frameW, frameH) / 4,
            bbox.width,
            bbox.height,
            landmarkTime);
        main_println(display, 10, frameH, 3, text);

        imshow(windowName, display);

        // check if need to wait some time to keep the FPS
        flag = (char)cvWaitKey((frameTime > processTime) ? frameTime - processTime + 1 : 1) != 27;
        //flag = (char)cvWaitKey(0) != 27;
        // write the output video
        if (writer != NULL) (*writer) << display;
    }

    // release resources
    delete detector;
    delete writer;
    delete device;
    cvDestroyWindow(windowName);
}
