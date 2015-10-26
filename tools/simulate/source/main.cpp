
#include <opencv2/opencv.hpp>
#include <face-detector/detector.hpp>
#include "../../../modules/face-landmark/source/ert/ShapePredictorTrainer.hh"
#include "../../../modules/face-landmark/source/ert/SampleList.hh"

#include <iostream>
#include <fstream>
#include <cstring>
#include <getopt.h>


#define FACE_CASCADE_FILE     "haarcascade_frontalface_alt2.xml"


using namespace ert;
using namespace std;
using namespace vasr::detector;


static string imageFileName = "";

static string modelFileName = "";

static bool useViolaJones = false;

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


class MainViewer : public ShapePredictorViewer
{

	public:
		MainViewer(
			const cv::Mat &image,
			const ObjectDetection &gold );

		~MainViewer() { };

		void show(
			int cascade,
			int tree,
			const ObjectDetection &current );

	private:
		const cv::Mat &image;
		cv::Mat display;
		const ObjectDetection &gold;

};


MainViewer::MainViewer(
	const cv::Mat &image,
	const ObjectDetection &gold ) : image(image), gold(gold)
{
}


void MainViewer::show(
	int cascade,
	int tree,
	const ObjectDetection &current )
{
	cvtColor(image, display, CV_GRAY2BGR);
	gold.plot(display, LAYOUT_68_PARTS, Scalar(255,0,0));
	current.plot(display, LAYOUT_68_PARTS, Scalar(0,0,255));

	std::stringstream ss;
	ss << "Tree #" << tree + 1 << " of cascade #" << cascade + 1;
	main_println(display, 0, 0, 0, ss.str());

	cv::imshow("Simulation", display);
	cv::waitKey(1);
}


bool main_load(
	const std::string &imageFileName,
	cv::Mat &image,
	ObjectDetection &annot )
{
	// load the image
#if (0)
	cv::cvtColor(imread(line), image, CV_BGR2GRAY);
#else
	cv::Mat temp[3];
	cv::split(imread(imageFileName), temp);
	temp[0].convertTo(temp[0], CV_32F);
	temp[1].convertTo(temp[1], CV_32F);
	temp[2].convertTo(temp[2], CV_32F);
	image = temp[0] + temp[1] + temp[2];
	image /= 3.0;
	image.convertTo(image, CV_8U);
#endif

	CascadeClassifier *faceCascade = new CascadeClassifier(FACE_CASCADE_FILE);
	ViolaJones *detector = new ViolaJones(faceCascade);

	Rect face;
	if (useViolaJones && !detector->detect(image, face))
		throw 1;

	// load the annotations
	std::string pointsFile = SampleList::changeExtension(imageFileName, "pts");
	try
	{
		annot.load(pointsFile);
		if (useViolaJones) annot.set_rect(face);
		return true;
	} catch (...)
	{
		std::cout << "   Ignoring file " << pointsFile << std::endl;
		return false;
	}
}


void main_usage()
{
    std::cerr << "Usage: tool_simulate -i <image file> -m <model file> [ -v ]" << std::endl << std::endl;
    std::cerr << "   -i  Image file name which face landmarks should be detected" << std::endl;
    std::cerr << "   -m  Existing model file name." << std::endl;
    std::cerr << "   -v  Use Viola-Jones face detector instead of computing the bounding" << std::endl;
    std::cerr << "       using the part annotations (rectangle covering all points plus" << std::endl;
    std::cerr << "       10% border)." << std::endl;
    exit(EXIT_FAILURE);
}


void main_parseOptions( int argc, char **argv )
{
    int opt;

    while ((opt = getopt(argc, argv, "i:m:v")) != -1)
    {
        switch (opt)
        {
            case 'i':
                imageFileName = string(optarg);
                break;
            case 'm':
				modelFileName = string(optarg);
				break;
            case 'v':
				useViolaJones = true;
				break;
            default: /* '?' */
                main_usage();
        }
    }
    if (imageFileName.empty() || modelFileName.empty())
    {
		main_usage();
	}
}

int main(int argc, char** argv)
{
	main_parseOptions(argc, argv);


	try
	{
		// load the shape model from file
		ShapePredictor model;
		std::ifstream input(modelFileName.c_str());
		model.deserialize(input);
		input.close();

		cv::Mat image;
		ObjectDetection annot;
		main_load(imageFileName, image, annot);

		MainViewer viewer(image, annot);

		ObjectDetection det = model.detect(image, annot.get_rect(), &viewer);

		char key = 0;
		while (key != 'q') key = cv::waitKey();

	} catch (exception& e)
	{
		cout << "Exception thrown!" << endl;
		cout << e.what() << endl;
	}
}
