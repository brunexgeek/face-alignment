#include <cstring>
#include <stdlib.h>
#include <cv.h>
#include <cvaux.h>
#include <opencv2/opencv.hpp>
#include <getopt.h>
#include <dirent.h>
#include <cmath>


using namespace cv;


#define MAX_LONG_EDGE         1000.0

#define INFOBOX_HEIGHT        100

#define FRAME_MAX_WIDTH       640

#define FACE_CASCADE_FILE     "haarcascade_frontalface_alt.xml"

#define FLANDMARK_FILE        "flandmark_model.dat"


const char *imageFileName = NULL;
const char *pointsFileName = NULL;
const char *fittedFileName = NULL;
const char *directory = NULL;


void main_usage()
{
    std::cerr << "Usage: tool_viewer -i <image file> -f <fitted file> -p <points file>\n";
    std::cerr << "Usage: tool_viewer -d <files path>\n";
    exit(EXIT_FAILURE);
}


void main_parseOptions( int argc, char **argv )
{
    int opt;

    while ((opt = getopt(argc, argv, "i:p:f:d:")) != -1)
    {
        switch (opt)
        {
            case 'i':
                imageFileName = optarg;
                break;
            case 'p':
                pointsFileName = optarg;
                break;
            case 'f':
                fittedFileName = optarg;
                break;
            case 'd':
				directory = optarg;
				break;
            default: /* '?' */
                main_usage();
        }
    }
    if (directory == NULL && (imageFileName == NULL || pointsFileName == NULL || fittedFileName == NULL))
    {
		main_usage();
	}
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


std::vector<Point> *loadPointsFile( const char *fileName )
{
	char *line;
	size_t len = 0;
	float x, y;
	int lines = 0;
	FILE *fp;
	int p;
	std::vector<Point> *points = NULL;
	char s[32];

	fp = fopen(fileName, "rt");
	if (fp == NULL) return 0;

	while(!feof(fp))
	{
		if ( getline(&line, &len, fp) < 0) continue;

		if ( points == NULL && strstr(line, "n_points:") != NULL )
		{
			if (sscanf(line, "%s %i", s, &p) == 2)
				points = new std::vector<Point>(p);
		}

		if ( points != NULL && sscanf(line, "%f %f", &x, &y) == 2)
		{
			(*points)[lines] = Point(x, y);;
			lines++;
		}
	}
	fclose(fp);

	return points;
}


Point2f operator*(cv::Mat M, const cv::Point2f& p)
{
	cv::Mat src(3/*rows*/,1 /* cols */,CV_64F);

	src.at<double>(0,0)=p.x;
	src.at<double>(1,0)=p.y;
	src.at<double>(2,0)=1.0;

	cv::Mat dst = M*src; //USE MATRIX ALGEBRA
	return cv::Point2f(dst.at<double>(0,0),dst.at<double>(1,0));
}


Rect *main_computeBoundingBox(
	std::vector<Point> &points,
	float border )
{
	float x, y;
	float minX = 100000, minY = 100000, maxX = -100, maxY = -100;
	Rect *bbox = new Rect(0, 0, 0, 0);

	for (size_t i = 0; i < points.size(); ++i)
	{
		x = points[i].x;
		y = points[i].y;
		if (minX > x) minX = x;
		if (minY > y) minY = y;
		if (maxX < x) maxX = x;
		if (maxY < y) maxY = y;
	}

	bbox->x = minX;
	bbox->y = minY;
	bbox->width = maxX - minX;
	bbox->height = maxY - minY;

	bbox->x -= (float(bbox->width) * border);
	bbox->y -= (float(bbox->height) * border);
	bbox->width += (float(bbox->width) * (border * 2));
	bbox->height += (float(bbox->height) * (border * 2));

	printf("{ (%d, %d) (%d, %d) }\n", bbox->x, bbox->y, bbox->width, bbox->height);

	return bbox;
}


void main_plotFace(
	Mat &image,
	std::vector<Point> *points,
	const Scalar &color )
{
	if (points != NULL)
	{
		// contorno da face
		for (size_t i = 0; i < 16; ++i)
			cv::line(image, (*points)[i], (*points)[i+1], color, 1);
		// sobrancelha esquerda
		for (size_t i = 17; i < 21; ++i)
			cv::line(image, (*points)[i], (*points)[i+1], color, 1);
		// sonbrancelha direita
		for (size_t i = 22; i < 26; ++i)
			cv::line(image, (*points)[i], (*points)[i+1], color, 1);
		// linha vertical do nariz
		for (size_t i = 27; i < 30; ++i)
			cv::line(image, (*points)[i], (*points)[i+1], color, 1);
		// linha horizontal do nariz
		for (size_t i = 31; i < 35; ++i)
			cv::line(image, (*points)[i], (*points)[i+1], color, 1);
		// olho esquerdo
		for (size_t i = 36; i < 41; ++i)
			cv::line(image, (*points)[i], (*points)[i+1], color, 1);
		cv::line(image, (*points)[41], (*points)[36], color, 1);
		// olho direito
		for (size_t i = 42; i < 47; ++i)
			cv::line(image, (*points)[i], (*points)[i+1], color, 1);
		cv::line(image, (*points)[47], (*points)[42], color, 1);
		// parte externa da boca
		for (size_t i = 48; i < 59; ++i)
			cv::line(image, (*points)[i], (*points)[i+1], color, 1);
		cv::line(image, (*points)[59], (*points)[48], color, 1);
		// parte externa da boca
		for (size_t i = 60; i < 67; ++i)
			cv::line(image, (*points)[i], (*points)[i+1], color, 1);
		cv::line(image, (*points)[67], (*points)[60], color, 1);
	}
}


char main_display(
	const char *imageFile,
	const char *pointsFile,
	const char *fittedFile )
{
	char key;
	int longEdge;
	float factor = 1;
	size_t i;

	std::cout << "Displaying " << imageFile << std::endl;

	Mat image = imread(imageFile);
	std::vector<Point> *points = loadPointsFile(pointsFile);
	std::vector<Point> *fitted = loadPointsFile(fittedFile);

	//longEdge = std::max(image.rows, image.cols);
	longEdge = image.rows;
	if (longEdge > MAX_LONG_EDGE)
	{
		factor = (float)longEdge / MAX_LONG_EDGE;
		std::cout << "Adjusting to factor " << factor << std::endl;
		resize(image, image,
			cv::Size( round( (float)image.cols / factor ), round( (float)image.rows / factor ) ) );

		for (i = 0; points != NULL && i < points->size(); ++i)
		{
			(*points)[i].x = round( (float)(*points)[i].x / factor );
			(*points)[i].y = round( (float)(*points)[i].y / factor );
		}

		for (i = 0; fitted != NULL && i < fitted->size(); ++i)
		{
			(*fitted)[i].x = round( (float)(*fitted)[i].x / factor );
			(*fitted)[i].y = round( (float)(*fitted)[i].y / factor );
		}
	}

	main_plotFace(image, points, Scalar(255, 0, 0));
	main_plotFace(image, fitted, Scalar(0, 0, 255));

	Rect *bbox = main_computeBoundingBox(*points, 0.2);
	if (bbox != NULL)
	{
		cv::rectangle(image, *bbox, Scalar(0, 255, 0));
	}

    imshow("Image", image);
    while (1)
    {
		key = waitKey(0);
		if (key == 'q' || key == ',' || key == '.') break;
	}

	return key;
}


std::vector<string> *main_listDirectory(
	const char *directory )
{
	struct dirent *entry;
	std::vector<string> *list = new std::vector<string>();

	DIR *dir = opendir(directory);
	if (dir == NULL) return NULL;

	while ((entry = readdir(dir)) != NULL)
	{
		if (strstr(entry->d_name, ".jpg") == NULL) continue;

		entry->d_name[ strlen(entry->d_name) - 4 ] = 0;
		list->push_back(entry->d_name);
	}

	return list;
}


int main( int argc, char** argv )
{
    main_parseOptions(argc, argv);

	if (directory != NULL)
	{
		std::vector<string> *files = main_listDirectory(directory);
		if (files != NULL)
		{
			int index = 0;
			while (1)
			{
				string fileName = string(directory) + "/" + files->at(index);
				string imageFile = fileName + ".jpg";
				string pointsFile = fileName + ".pts";
				string fittedFile = fileName + ".fit";
				char key = main_display(imageFile.c_str(), pointsFile.c_str(), fittedFile.c_str());
				if (key == ',')
				{
					--index;
					if (index < 0) index = 0;
					continue;
				}
				if (key == '.')
				{
					++index;
					if (index >= (int)files->size()) index = files->size() - 1;
					continue;
				}
				if (key == 'q') break;
			}
			delete files;
		}
	}
	else
	{
		main_display(imageFileName, pointsFileName, fittedFileName);
	}

}
