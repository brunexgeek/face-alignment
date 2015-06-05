/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Adapted from "example_2" of "flandmark".
 */

#include <cstring>
#include <stdlib.h>
#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <face-detector/detector.hpp>
#include <face-landmark/landmark.hpp>


using namespace vasr::detector;
using namespace vasr::landmark;


#define INFOBOX_HEIGHT        100

#define FRAME_MAX_WIDTH       640


IplImage* getCameraFrame(CvCapture* &camera, const char *filename = 0, int camid=0, int width=320, int height=240)
{
    IplImage *frame = 0;
    int w, h;

    // If the camera hasn't been initialized, then open it.
    if (!camera)
    {
        if (!filename)
        {
            printf("Acessing the camera ...\n");
            camera = cvCaptureFromCAM(camid);
            // Try to set the camera resolution.
            cvSetCaptureProperty(camera, CV_CAP_PROP_FRAME_WIDTH, width);
            cvSetCaptureProperty(camera, CV_CAP_PROP_FRAME_HEIGHT, height);
        } else {
            printf("Acessing the video sequence ...\n");
            camera = cvCaptureFromAVI(filename);
        }

        if (!camera)
        {
            printf("Couldn't access the camera.\n");
            exit(1);
        }

        // Get the first frame, to make sure the camera is initialized.
        frame = cvQueryFrame( camera );
        //frame = cvRetrieveFrame(camera);

        if (frame)
        {
            w = frame->width;
            h = frame->height;
            printf("Got the camera at %dx%d resolution.\n", w, h);
        }
        //sleep(10);
    }

    // Wait until the next camera frame is ready, then grab it.
    frame = cvQueryFrame( camera );
    //frame = cvRetrieveFrame(camera);
    if (!frame)
    {
        printf("Couldn't grab a camera frame.\n");
        return frame;
    }
    return frame;
}



bool detector_landmark(
    IplImage *orig,
    IplImage* input,
    FaceDetector *detector,
    FaceLandmark *landmarker,
    CvRect *bbox )
{
    if (!detector->detect(input, bbox))
    {
        printf("NO Face\n");
        return false;
    }

    double t = (double)cvGetTickCount();

    landmarker->detect(input, bbox);

    t = (double)cvGetTickCount() - t;
    int ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );

    printf("Detection of facial landmark on all faces took %d ms\n", ms);
    return true;
}


static void main_println( IplImage *image, CvFont *font, int x, int y, int line, const char *text )
{
    CvSize size;
    int baseline;

    cvGetTextSize(text, font, &size, &baseline);
    cvPutText(image, text, cvPoint(x, y + (line + 1) * (size.height + 10)), font, cvScalar(255, 255, 255, 0));
}


int main( int argc, char** argv )
{
    char flandmark_window[] = "flandmark_example2";
    double startTime, processTime, finalTime, frameTime;
    int ms;
    bool mustResize = false, hasFace;
    ViolaJones *detector;
    FaceLandmark *landmarker;

    const char *infname = 0;
    const char *outfname = 0;
    bool video = false, savevideo = false;

    CvVideoWriter *writer = 0;
    int vidfps = 0, frameW = 0, frameH = 0, nframes = 0;

    CvCapture* camera = 0;  // The camera device.
    IplImage *frame = 0;

    if (argc == 1)
    {
        exit(1);
    }

    if (argc > 1)
    {
        infname = argv[1];
        printf("infname = %s\n", infname);
        video = (strlen(infname) > 1) ? true : false;

        if (video)
        {
            frame = getCameraFrame(camera, infname);
            frameH = (int)cvGetCaptureProperty(camera, CV_CAP_PROP_FRAME_HEIGHT);
            frameW = (int)cvGetCaptureProperty(camera, CV_CAP_PROP_FRAME_WIDTH);
            nframes = (int)cvGetCaptureProperty(camera, CV_CAP_PROP_FRAME_COUNT);
            vidfps = (int)cvGetCaptureProperty(camera, CV_CAP_PROP_FPS);
        }
        else
        {
            int width=320, height=240, camid;
            camid = ::atoi(argv[1]);
            if (argc > 4)
            {
                width = ::atoi(argv[3]);
                height = ::atoi(argv[4]);
            }
            frame = getCameraFrame(camera, 0, camid, width, height);
            vidfps = 10;
            frameW = 640;
            frameH = 480;
        }
    }
    frameTime = 1000 / vidfps;

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
    cvNamedWindow(flandmark_window, 0);
    int windowW = frameW;
    int windowH = frameH + INFOBOX_HEIGHT;
    cvResizeWindow(flandmark_window, windowW, windowH);

    if (argc > 2)
    {
        outfname = argv[2];
        savevideo = true;
        //writer = cvCreateAVIWriter(outfname, fourcc, vidfps, cvSize(frameW, frameH));
        writer = cvCreateVideoWriter(outfname, CV_FOURCC('M', 'J', 'P', 'G'), vidfps, cvSize(windowW, windowH));
    }

    // Haar Cascade file, used for Face Detection.
    char faceCascadeFilename [] = "haarcascade_frontalface_alt.xml";
    // Load the HaarCascade classifier for face detection.
    CvHaarClassifierCascade* faceCascade;
    faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename, 0, 0, 0);
    if( !faceCascade )
    {
        printf("Couldnt load Face detector '%s'\n", faceCascadeFilename);
        exit(1);
    }

    // ------------- begin flandmark load model
    startTime = (double)cvGetTickCount();
    FLANDMARK_Model * model = flandmark_init("flandmark_model.dat");
    if (model == 0)
    {
        printf("Structure model was not created. Corrupted file flandmark_model.dat?\n");
        exit(1);
    }
    processTime = (double)cvGetTickCount() - startTime;
    ms = cvRound( processTime / ((double)cvGetTickFrequency() * 1000.0) );
    printf("Structure model loaded in %d ms.\n", ms);
    // ------------- end flandmark load model

    CvRect bbox;
    double *landmarks = (double*)malloc(2*model->data.options.M*sizeof(double));
    IplImage *frame_bw = cvCreateImage(cvSize(frameW, frameH), IPL_DEPTH_8U, 1);

    IplImage *display = cvCreateImage(cvSize(windowW, windowH), IPL_DEPTH_8U, 3);

    char text[256];
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 0.5, CV_AA);

    int frameid = 0;
    bool flag = true;

    detector = new ViolaJones(faceCascade);
    landmarker = new FLandmark(model);

    if (video)
    {
        while (flag)
        {
            if (++frameid >= nframes-2) break;

            startTime = (double)cvGetTickCount();

            frame = getCameraFrame(camera, infname);
            if (!frame) break;

            // check if need to resize the frame
            if (mustResize)
            {
                cvResize(frame, tempFrame, cv::INTER_AREA);
                frame = tempFrame;
            }
            // convert the original frame to grayscale and look for landmarks
            cvConvertImage(frame, frame_bw);
            hasFace = detector_landmark(frame, frame_bw, detector, landmarker, &bbox);
            cvConvertImage(frame_bw, frame);

            // display landmarks
            if (hasFace)
            {
                cvRectangle(frame, cvPoint(bbox.x, bbox.y), cvPoint(bbox.x+bbox.width, bbox.y+bbox.height), CV_RGB(255,0,0) );
                cvRectangle(frame, cvPoint(model->bb[0], model->bb[1]), cvPoint(model->bb[2], model->bb[3]), CV_RGB(0,0,255) );
                //cvCircle(frame, cvPoint((int)landmarks[0], (int)landmarks[1]), 3, CV_RGB(0, 0,255), CV_FILLED);
                CvPoint point;
                for (int i = 1; i < model->data.options.M; ++i)
                {
                    landmarker->getLandmark(i, point);
                    cvCircle(frame, point, 3, CV_RGB(255,0,0), CV_FILLED);
                }
            }

            // update the display image
            cvCopyMakeBorder(frame, display, cvPoint(0, 0), cv::BORDER_CONSTANT, cvScalar(0.0f, 0.0f, 0.0f, 0.0f));

            processTime = ((double)cvGetTickCount() - startTime) / ((double)cvGetTickFrequency() * 1000);
            finalTime = processTime + ((frameTime > processTime) ? frameTime - processTime : 0);

            // print FPS
            sprintf(text, "Original: %d fps    Displayed: %.2f fps    Processed: %.2f fps",
                vidfps,
                1000.0 / finalTime,
                1000.0 / processTime);
            main_println(display, &font, 10, frameH, 0, text);
            // print frame count
            sprintf(text, "Frame %d of %d", frameid, nframes);
            main_println(display, &font, 10, frameH, 1, text);

            cvShowImage(flandmark_window, display );

            // check if need to wait some time to keep the FPS
            flag = (char)cvWaitKey((frameTime > processTime) ? frameTime - processTime + 1 : 1) != 27;

            if (savevideo)
            {
                cvWriteFrame(writer, display);
            }
        }
    } else {
        while ( (char)cvWaitKey(20) != 27 )
        {
            startTime = (double)cvGetTickCount();
            // Quit on "Escape" key.
            frame = video ? getCameraFrame(camera, infname) : getCameraFrame(camera);

            cvConvertImage(frame, frame_bw);
            hasFace = detector_landmark(frame, frame_bw, detector, landmarker, &bbox);

            // display landmarks
            if (hasFace)
            {
                cvRectangle(frame, cvPoint(bbox.x, bbox.y), cvPoint(bbox.x+bbox.width, bbox.y+bbox.height), CV_RGB(255,0,0) );
                cvRectangle(frame, cvPoint(model->bb[0], model->bb[1]), cvPoint(model->bb[2], model->bb[3]), CV_RGB(0,0,255) );
                //cvCircle(frame, cvPoint((int)landmarks[0], (int)landmarks[1]), 3, CV_RGB(0, 0,255), CV_FILLED);
                for (int i = 2; i < 2*model->data.options.M; i += 2)
                {
                    cvCircle(frame, cvPoint(int(landmarks[i]), int(landmarks[i+1])), 3, CV_RGB(255,0,0), CV_FILLED);
                }
            }

            processTime = (double)cvGetTickCount() - startTime;
            sprintf(text, "%.2f fps", (1000.0*1000.0*(double)cvGetTickFrequency())/processTime );
            //cvPutText(frame, text, cvPoint(10, 40), &font, cvScalar(255, 0, 0, 0));
            main_println(display, &font, 10, frameH, 0, text);

            cvShowImage(flandmark_window, frame);

            if (savevideo)
            {
                cvWriteFrame(writer, frame);
            }
        }
    }

    if (tempFrame != NULL)
        cvReleaseImage(&tempFrame);

    delete detector;

    // Free the camera.
    free(landmarks);
    cvReleaseCapture(&camera);
    cvReleaseHaarClassifierCascade(&faceCascade);
    cvDestroyWindow(flandmark_window);
    flandmark_free(model);
}
