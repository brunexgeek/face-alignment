#include <opencv2/opencv.hpp>
#include "../../../modules/face-landmark/source/ert/shape_predictor.h"
#include <iostream>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>

using namespace std;
using namespace cv;


char imageFileName[256];
char pointFileName[256];
char outputFileName[256];
char temp[64];


// ----------------------------------------------------------------------------------------


uint8_t computeBoundingBox(
	const char *fileName,
	float *bbox,
	float border )
{
	char *line;
	size_t len = 0;
	float x, y;
	int lines = 0;
	float minX = 100000, minY = 100000, maxX = -100, maxY = -100;
	FILE *fp;

	fp = fopen(fileName, "rt");
	if (fp == NULL) return 0;

	while(!feof(fp))
	{
		if ( getline(&line, &len, fp) < 0) continue;
		if ( sscanf(line, "%f %f", &x, &y) == 2 )
		{
			if (minX > x) minX = x;
			if (minY > y) minY = y;
			if (maxX < x) maxX = x;
			if (maxY < y) maxY = y;
			lines++;
		}
	}
	fclose(fp);

	if (lines != 68) return 0;

	bbox[0] = minY;
	bbox[1] = minX;
	bbox[2] = maxY - minY;
	bbox[3] = maxX - minX;

	bbox[0] -= (float(bbox[2]) * border);
	bbox[1] -= (float(bbox[3]) * border);
	bbox[2] += (float(bbox[2]) * (border * 2));
	bbox[3] += (float(bbox[3]) * (border * 2));

	return 1;
}

#if (0)
void loadBoundingBoxes( const char *fileName, float*** values, int *count )
{
	int lines = 0;
	size_t len = 0;
	FILE *fp;
	char *content;
	char ch;

	fp = fopen(fileName, "rt");
	if (fp == NULL) return NULL;

	while(!feof(fp))
	{
		ch = fgetc(fp);
		if(ch == '\n') lines++;
	}
	fseek(fp, 0, SEEK_SET);

	float ** data = (float**) calloc(lines, sizeof(float*));

	lines = 0;
	while(!feof(fp))
	{
		if ( getline(&content, &len, fp) < 0) continue;
		data[lines] = (float*) calloc(4, sizeof(float));
		sscanf(content, "%f %f %f %f",
			&data[lines][0],
			&data[lines][1],
			&data[lines][2],
			&data[lines][3]);
		printf("%f %f %f %f\n",
			data[lines][0],
			data[lines][1],
			data[lines][2],
			data[lines][3]);
		lines++;
	}

	return data;
}
#endif


Point2f operator*(cv::Mat M, const cv::Point2f& p)
{
	cv::Mat src(3/*rows*/,1 /* cols */,CV_64F);

	src.at<double>(0,0)=p.x;
	src.at<double>(1,0)=p.y;
	src.at<double>(2,0)=1.0;

	cv::Mat dst = M*src; //USE MATRIX ALGEBRA
	return cv::Point2f(dst.at<double>(0,0),dst.at<double>(1,0));
}


int main(int argc, char** argv)
{
	struct dirent *entry;
	DIR *dir;
	int pos;
	float data[4];
	int i = 0;

	//loadBoundingBoxes(argv[3], &data, &files);
	//return 0;

	dir = opendir(argv[1]);
	if (dir == NULL) return 1;

    try
    {
        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.
        dlib::ShapePredictor sp;
        //deserialize("sp.dat") >> sp;

        // Loop over all the images provided on the command line.
        while ((entry = readdir(dir)) != NULL)
        {
			if (entry->d_type != DT_REG) continue;
			if (strstr(entry->d_name, ".pts") != NULL) continue;
			if (strstr(entry->d_name, ".fit") != NULL) continue;

            cout << "Processing image " << entry->d_name << endl;

			sprintf(imageFileName, "%s/%s", argv[1], entry->d_name);

			strncpy(temp, entry->d_name, sizeof(temp)-1);
			for (pos = 0; temp[pos] != 0; ++pos)
			{
				if (temp[pos] == '.')
				{
					temp[pos] = 0;
					break;
				}
			}
			sprintf(outputFileName, "%s/%s.fit", argv[1], temp);
			sprintf(pointFileName, "%s/%s.pts", argv[1], temp);

            Mat img;
            img = cv::imread(imageFileName);
            // Make the image larger so we can detect small faces.
            //pyramid_up(images_test[i]);

			computeBoundingBox(pointFileName, data, 0.1);
			Rect rect( data[1], data[0], data[1] + data[3], data[0] + data[2]);
			printf("{ (%d, %d) (%d, %d) }\n", int(data[1]), int(data[0]), int(data[3]), int(data[2]));
            // Now tell the face detector to give us a list of bounding boxes
            // around all the faces in the image.
            //std::vector<rectangle> dets = detector(img);
            //cout << "Number of faces detected: " << dets.size() << endl;

            // Now we will go ask the shape_predictor to tell us the pose of
            // each face we detected.
            for (unsigned long j = 0; j < 1; ++j)
            {
                dlib::FullObjectDetection shape = sp.detect(img, rect);
                // You get the idea, you can get all the face part locations if
                // you want them.  Here we just store them in shapes so we can
                // put them on the screen.
                FILE *output;

                output = fopen(outputFileName, "wt");
                if (output != NULL)
                {
					fprintf(output, "version: 1\nn_points:  %ld\n{\n", shape.num_parts());
					for (unsigned long c = 0; c < shape.num_parts(); ++c)
					{
						fprintf(output, "%0.2f %0.2f\n",
							(float)shape.part(c).x,
							(float)shape.part(c).y );
					}
					fprintf(output, "}");
					fclose(output);
				}

            }
        }
        closedir(dir);
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

