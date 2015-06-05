#include <cstring>
#include <stdlib.h>
#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <getopt.h>
#include <face-detector/detector.hpp>
#include <face-landmark/landmark.hpp>


using namespace vasr::detector;
using namespace vasr::landmark;


#define INFOBOX_HEIGHT        100

#define FRAME_MAX_WIDTH       640

#define FACE_CASCADE_FILE     "haarcascade_frontalface_alt.xml"

#define FLANDMARK_FILE        "flandmark_model.dat"


bool useCamera = false;
bool showProcessedFrame = false;
char *inputFileName = NULL;
char *outputFileName = NULL;
bool useRotation = false;


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
    CvCapture **device,
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
            *device = cvCaptureFromCAM(camera);
            cvSetCaptureProperty(*device, CV_CAP_PROP_FRAME_WIDTH, width);
            cvSetCaptureProperty(*device, CV_CAP_PROP_FRAME_HEIGHT, height);
        }
        else
        {
            printf("Capturing from file '%s'\n", fileName);
            *device = cvCaptureFromAVI(fileName);
        }

        if (*device == NULL)
        {
            printf("Error connecting to capture device\n");
            return 1;
        }
    }
    return 0;
}



bool main_landmark(
    IplImage *orig,
    IplImage* input,
    FaceDetector *detector,
    FaceLandmark *landmarker,
    CvRect *bbox,
    int *time )
{
    static bool hasFace = false;

    if (!detector->detect(input, bbox))
    {
        if (hasFace)
        {
            printf("No face detected!\n");
            hasFace = false;
        }
        return false;
    }
    else
    {
        if (!hasFace)
        {
            printf("Face detected!\n");
            hasFace = true;
        }
    }

    double startTime = (double)cvGetTickCount();

    landmarker->detect(input, bbox);


    if (time != NULL)
    {
        startTime = (double)cvGetTickCount() - startTime;
        *time = cvRound( startTime / ((double)cvGetTickFrequency() * 1000.0) );
    }

    return true;
}


static void main_println( IplImage *image, CvFont *font, int x, int y, int line, const char *text )
{
    CvSize size;
    int baseline;

    cvGetTextSize(text, font, &size, &baseline);
    cvPutText(image, text, cvPoint(x, y + (line + 1) * (size.height + 10)), font, cvScalar(255, 255, 255, 0));
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


static cv::Mat main_getRotationMatrix(
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

    main_getDelta(left, right, dx, dy, &center);
    //printf("delta.x = %f  delta.y = %f\n", dx, dy);
    if (dx < 0) dx *= -1;

    double m = dy / dx;
    double angle = (m == 0) ? 0 : atan(m) * -180 / 3.14159265;
printf("Angle: %f degress gradient %f)\n", angle, m);
    return cv::getRotationMatrix2D(center, angle, 1);
}

cv::Point2f operator*(cv::Mat M, const cv::Point2f& p){
        cv::Mat src(3/*rows*/,1 /* cols */,CV_64F);

        src.at<double>(0,0)=p.x;
        src.at<double>(1,0)=p.y;
        src.at<double>(2,0)=1.0;

        cv::Mat dst = M*src; //USE MATRIX ALGEBRA
        return cv::Point2f(dst.at<double>(0,0),dst.at<double>(1,0));
}

static void main_rotateFace(
    IplImage *image,
    IplImage *rotated,
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
    lp.x += -1 * width * 0.1f;
    rp.x += width * 0.1f;

    width = rp.x - lp.x;
    float height = width * 0.5f;
    lp.y += -1 * height / 3;
    rp.y += 2 * (height / 3);

    cv::Mat src, dst;
    src = image;
    warpAffine(src, dst, rot, dst.size());
    *rotated = dst;

    cvRectangle(rotated, lp, rp, CV_RGB(255,0,0) );
}


int main( int argc, char** argv )
{
    double startTime, processTime, finalTime, frameTime;
    int result;
    bool mustResize = false;
    bool hasFace;
    ViolaJones *detector;
    FaceLandmark *landmarker;
    const char *windowName = "Visual ASR Demo";
    CvVideoWriter *writer = NULL;
    int inputFPS = 0;
    int frameW = 0, originalW;
    int frameH = 0, originalH;
    int frameCount = 0;
    CvCapture* device = 0;
    IplImage *frame = 0;

    main_parseOptions(argc, argv);

    // initialize the input device
    if (useCamera)
        result = main_initDevice(&device, NULL, 0, 640, 480);
    else
        result = main_initDevice(&device, inputFileName);
    if (result != 0) return 1;
    originalH = frameH = (int)cvGetCaptureProperty(device, CV_CAP_PROP_FRAME_HEIGHT);
    originalW = frameW = (int)cvGetCaptureProperty(device, CV_CAP_PROP_FRAME_WIDTH);
    frameCount = (int)cvGetCaptureProperty(device, CV_CAP_PROP_FRAME_COUNT);
    inputFPS = (int)cvGetCaptureProperty(device, CV_CAP_PROP_FPS);
    frameTime = 1000 / inputFPS;

    // limit the width of the input frame
    IplImage *tempFrame = NULL;
    if (frameW > FRAME_MAX_WIDTH)
    {
        frameH = (frameH * FRAME_MAX_WIDTH / frameW);
        frameW = FRAME_MAX_WIDTH;
        mustResize = true;
        tempFrame = cvCreateImage(cvSize(frameW, frameH), IPL_DEPTH_8U, 3);
        frame = tempFrame;
    }
    // create the display window
    cvNamedWindow(windowName, 0);
    int windowW = frameW;
    int windowH = frameH + INFOBOX_HEIGHT;
    cvResizeWindow(windowName, windowW, windowH);

    // initialize the output video writer
    if (outputFileName != NULL)
        writer = cvCreateVideoWriter(outputFileName,
            CV_FOURCC('M', 'J', 'P', 'G'), inputFPS,
            cvSize(windowW, windowH));

    // load the HaarCascade classifier for face detection
    CvHaarClassifierCascade* faceCascade;
    faceCascade = (CvHaarClassifierCascade*)cvLoad(FACE_CASCADE_FILE, 0, 0, 0);
    if (!faceCascade)
    {
        printf("Error loading face cascade file '%s'\n", FACE_CASCADE_FILE);
        return 1;
    }

    // load flandmark model
    FLANDMARK_Model * model = flandmark_init(FLANDMARK_FILE);
    if (model == 0)
    {
        printf("Error laoding flandmark model '%s'\n", FLANDMARK_FILE);
        return 1;
    }

    // create the temporary buffers
    CvRect bbox;
    IplImage *frame_bw = cvCreateImage(cvSize(frameW, frameH), IPL_DEPTH_8U, 1);
    IplImage *rotated = cvCreateImage(cvSize(frameW, frameH), IPL_DEPTH_8U, 1);
    IplImage *display = cvCreateImage(cvSize(windowW, windowH), IPL_DEPTH_8U, 3);
    // create the GUI font
    char text[256];
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 0.5, CV_AA);

    int currentFrame = 0;
    bool flag = true;
    int landmarkTime = 0;

    // initialize the face detector and face landmarker
    detector = new ViolaJones(faceCascade);
    landmarker = new FLandmark(model);

    while (flag)
    {
        if (useCamera == false && ++currentFrame >= frameCount - 2) break;

        startTime = (double)cvGetTickCount();

        //frame = main_getFrame(&device, infname);
        frame = cvQueryFrame(device);
        if (!frame) break;

        // check if need to resize the frame
        if (mustResize)
        {
            cvResize(frame, tempFrame, cv::INTER_AREA);
            frame = tempFrame;
        }
        // convert the original frame to grayscale and look for landmarks
        cvConvertImage(frame, frame_bw);
        hasFace = main_landmark(frame, frame_bw, detector, landmarker, &bbox, &landmarkTime);
        if (showProcessedFrame) cvConvertImage(frame_bw, frame);
        if (hasFace && useRotation)
        {
            main_rotateFace(frame, rotated, *landmarker);
            frame = rotated;
        }

        // display landmarks
        if (hasFace)
        {
            cvRectangle(frame, cvPoint(bbox.x, bbox.y), cvPoint(bbox.x+bbox.width, bbox.y+bbox.height), CV_RGB(255,0,0) );
            cvRectangle(frame, cvPoint(model->bb[0], model->bb[1]), cvPoint(model->bb[2], model->bb[3]), CV_RGB(0,0,255) );
            //cvCircle(frame, cvPoint((int)landmarks[0], (int)landmarks[1]), 3, CV_RGB(0, 0,255), CV_FILLED);
            if (!useRotation)
            {
                CvPoint point;
                for (int i = 1; i < model->data.options.M; ++i)
                {
                    point.x = landmarker->getX(i);
                    point.y = landmarker->getY(i);
                    cvCircle(frame, point, 3, CV_RGB(255,0,0), CV_FILLED);
                }
            }
        }

        // update the display image
        cvCopyMakeBorder(frame, display, cvPoint(0, 0), cv::BORDER_CONSTANT, cvScalar(0.0f, 0.0f, 0.0f, 0.0f));
        // update time counters
        processTime = ((double)cvGetTickCount() - startTime) / ((double)cvGetTickFrequency() * 1000);
        finalTime = processTime + ((frameTime > processTime) ? frameTime - processTime : 0);

        // print FPS
        sprintf(text, "FPS: Original=%d    Displayed=%.2f    Processed=%.2f",
            inputFPS,
            1000.0 / finalTime,
            1000.0 / processTime);
        main_println(display, &font, 10, frameH, 0, text);
        // print frame count
        sprintf(text, "Frame %d of %d", currentFrame, frameCount);
        main_println(display, &font, 10, frameH, 1, text);
        // print frame size
        sprintf(text, "Frame: Original=%dx%d    Processed=%dx%d",
            originalW,
            originalH,
            frameW,
            frameH);
        main_println(display, &font, 10, frameH, 2, text);
        // print face size
        sprintf(text, "Face: Minimum=%dx%d    Current=%dx%d    Time=%d ms",
            std::min(frameW, frameH) / 4,
            std::min(frameW, frameH) / 4,
            bbox.width,
            bbox.height,
            landmarkTime);
        main_println(display, &font, 10, frameH, 3, text);

        cvShowImage(windowName, display );

        // check if need to wait some time to keep the FPS
        flag = (char)cvWaitKey((frameTime > processTime) ? frameTime - processTime + 1 : 1) != 27;
        // write the output video
        if (writer != NULL) cvWriteFrame(writer, display);
    }

    // release resources
    if (tempFrame != NULL) cvReleaseImage(&tempFrame);
    delete detector;
    cvReleaseCapture(&device);
    cvReleaseHaarClassifierCascade(&faceCascade);
    cvDestroyWindow(windowName);
    flandmark_free(model);
}
