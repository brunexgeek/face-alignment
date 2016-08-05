#include <cstring>
#include <stdlib.h>
#include <cv.h>
#include <cvaux.h>
#include <opencv2/opencv.hpp>
#include <getopt.h>
#include <dirent.h>
#include <cmath>
#include <face-detector/detector.hpp>
#include "../../../modules/face-landmark/source/ert/ObjectDetection.hh"
#include "../../../modules/face-landmark/source/ert/SampleList.hh"

using namespace cv;
using namespace ert;
using namespace vasr::detector;
using namespace std;


#define MAX_LONG_EDGE         800.0

#define INFOBOX_HEIGHT        100

#define FRAME_MAX_WIDTH       640

#define FACE_CASCADE_FILE     "haarcascade_frontalface_alt2.xml"

#define FLANDMARK_FILE        "flandmark_model.dat"


const char *imageFileName = NULL;


const char *pointsFileName = NULL;

const char *fittedFileName = NULL;

const char *directory = NULL;

bool showGoldParts = true;

bool showFittedParts = true;

bool useViolaJones = false;


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

void main_usage()
{
    std::cerr << "Usage: tool_viewer -i <image file> -f <fitted file> -p <points file>\n";
    std::cerr << "Usage: tool_viewer -d <files path>\n";
    exit(EXIT_FAILURE);
}


void main_parseOptions( int argc, char **argv )
{
    int opt;

    while ((opt = getopt(argc, argv, "i:p:f:d:v")) != -1)
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
			case 'v':
				useViolaJones = true;
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


static void main_println( Mat &image, int x, int y, int line, const std::string &text )
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


char main_display(
	const std::string &imageFile,
	const std::string &pointsFile,
	const std::string &fittedFile )
{
	char key;
	int longEdge;
	float factor = 1;
	bool violaJonesFail = false;

	std::cout << "Displaying " << imageFile << std::endl;

	Mat image = imread(imageFile);
	ObjectDetection gold(pointsFile);
	ObjectDetection fitted(fittedFile);
	Rect area, face;

	if (useViolaJones)
	{
		CascadeClassifier *faceCascade = new CascadeClassifier(FACE_CASCADE_FILE);
		ViolaJones *detector = new ViolaJones(faceCascade);

		Mat gray;
		cvtColor(image, gray, CV_BGR2GRAY);

		if (!detector->detect(gray, face))
		{
			std::cout << "Viola-Jones failed!" << std::endl;
			violaJonesFail = true;
		}
		else
			area = face;

		delete detector;
		delete faceCascade;
	}

	if (violaJonesFail || !useViolaJones)
	{
		// compute the ROI
		area = face = gold.get_rect();
#if (0)
		float border = std::max( (area.width * 0.1), (area.height * 0.1) );
		area.x      -= border;
		area.width  += border * 2;
		area.y      -= border;
		area.height += border * 2;
#endif
	}
#if (0)
	// adjust the ROI to fit in the image
	if (area.x < 0) area.x = 0;
	if (area.y < 0) area.y = 0;
	if (area.x + area.width > image.cols) area.width = image.cols - area.x;
	if (area.y + area.height > image.rows) area.height = image.rows - area.y;
	// crop the original image
	image = image(area);
	gold -= Point2f( area.x, area.y );
	fitted -= Point2f( area.x, area.y );
	// adjust the bouding box position
	face.x -= area.x;
	face.y -= area.y;
#endif
	//longEdge = std::max(image.rows, image.cols);
	longEdge = image.rows;
	if (longEdge > MAX_LONG_EDGE)
	{
		factor = (float)longEdge / MAX_LONG_EDGE;
		std::cout << "Adjusting to factor " << factor << std::endl;
		resize(image, image,
			cv::Size( round( (float)image.cols / factor ), round( (float)image.rows / factor ) ) );

		gold /= factor;
		fitted /= factor;
		face.x /= factor;
		face.y /= factor;
		face.width /= factor;
		face.height /= factor;
	}

	// plot the parts
	if (showGoldParts)
		gold.plot(image, LAYOUT_68_PARTS, Scalar(255, 0, 0));
	if (showFittedParts)
		fitted.plot(image, LAYOUT_68_PARTS, Scalar(0, 0, 255));
	// plot an rectangle to show the bouding box
	cv::rectangle(image, face, Scalar(0, 255, 0));

	main_println(image, 0, 0, 0, imageFile);

    imshow("Image", image);
    while (1)
    {
		key = waitKey(0);
		if (key == 'q' || key == ',' || key == '.' || key == 'g' || key == 'f') break;
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
		if (strstr(entry->d_name, ".jpg") == NULL &&
		    strstr(entry->d_name, ".png") == NULL) continue;

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
				string imageFile = string(directory) + "/" + files->at(index);
				string pointsFile = SampleList::changeExtension(imageFile, "pts");
				string fittedFile = SampleList::changeExtension(imageFile, "fit");
				char key = main_display(imageFile, pointsFile, fittedFile);
				switch (key)
				{
					case ',':
						--index;
						if (index < 0) index = 0;
						continue;
					case '.':
						++index;
						if (index >= (int)files->size()) index = files->size() - 1;
						continue;
					case 'f':
						showFittedParts = !showFittedParts;
						break;
					case 'g':
						showGoldParts = !showGoldParts;
						break;
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
