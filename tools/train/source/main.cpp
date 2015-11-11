
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


static string trainScriptFileName = "";

static string evaluateScriptFileName = "";

static string modelFileName = "";

static bool useAbsoluteScore = false;

static bool useViolaJones = false;

static int configTreeDepth = 0;

static int configTestSplits = 0;


class MainSampleLoader : public SampleLoader
{

	public:
		MainSampleLoader( bool useViolaJones );

		~MainSampleLoader();

		void load(
			const std::string &imageFileName,
			cv::Mat &image,
			ObjectDetection &annot );

	private:
		ViolaJones *detector;
		CascadeClassifier *faceCascade;
		bool useViolaJones;

};


MainSampleLoader::MainSampleLoader( bool useViolaJones )
{
	this->useViolaJones = useViolaJones;
	faceCascade = new CascadeClassifier(FACE_CASCADE_FILE);
	detector = new ViolaJones(faceCascade);
}

MainSampleLoader::~MainSampleLoader()
{
	delete detector;
	delete faceCascade;
}


void MainSampleLoader::load(
	const std::string &imageFileName,
	cv::Mat &image,
	ObjectDetection &annot )
{
	//static float originalSize = 0, croppedSize = 0;

	// load the image
	cv::Mat source = imread(imageFileName);
	cv::Mat gray;
#if (0)
	cv::cvtColor(source, gray, CV_BGR2GRAY);
#else
	cv::Mat temp[3];
	cv::split(source, temp);
	temp[0].convertTo(temp[0], CV_32F);
	temp[1].convertTo(temp[1], CV_32F);
	temp[2].convertTo(temp[2], CV_32F);
	gray = temp[0] + temp[1] + temp[2];
	gray /= 3.0;
	gray.convertTo(gray, CV_8U);
#endif
	source.release();

	Rect face;
	if (useViolaJones && !detector->detect(gray, face))
		throw 1;

	// load the annotations
	std::string pointsFile = SampleList::changeExtension(imageFileName, "pts");
	try
	{
		annot.load(pointsFile);
		if (useViolaJones) annot.set_rect(face);
		// remove the first 17 points
		annot.remove(0, 16);
	} catch (...)
	{
		std::cout << "   Ignoring file " << pointsFile << std::endl;
	}

	// crop the original image to save memory
	if (!useViolaJones)
	{
		Rect area = annot.get_rect();

		// adjust the ROI to fit in the image
		if (area.x < 0) area.x = 0;
		if (area.y < 0) area.y = 0;
		if (area.x + area.width > gray.cols) area.width = gray.cols - area.x;
		if (area.y + area.height > gray.rows) area.height = gray.rows - area.y;
		// crop the original image
		gray = gray(area);
		gray.copyTo(image);
		annot -= Point2f( area.x, area.y );
	}
	else
		image = gray;
}


double interocular_distance (
    const ObjectDetection& det )
{
    Point2f l, r;
    double cnt = 0;
    // Find the center of the left eye by averaging the points around
    // the eye.
    for (unsigned long i = 36; i <= 41; ++i)
    {
        l += det.part(i);
        ++cnt;
    }
    l = l / cnt;

    // Find the center of the right eye by averaging the points around
    // the eye.
    cnt = 0;
    for (unsigned long i = 42; i <= 47; ++i)
    {
        r += det.part(i);
        ++cnt;
    }
    r = r / cnt;

    // Now return the distance between the centers of the eyes

    return cv::norm(l-r);
}


std::vector<std::vector<double> > get_interocular_distances (
    const std::vector<std::vector<ert::ObjectDetection*> >& objects )
{
    std::vector<std::vector<double> > temp(objects.size());
    for (unsigned long i = 0; i < objects.size(); ++i)
    {
        for (unsigned long j = 0; j < objects[i].size(); ++j)
        {
            temp[i].push_back(interocular_distance(*objects[i][j]));
        }
    }
    return temp;
}


void main_usage()
{
    std::cerr << "Usage: tool_train -t <script file> -m <model file> [ -v -a ]" << std::endl;
    std::cerr << "       tool_train -e <script file> -m <model file> [ -v -a ]" << std::endl << std::endl;
    std::cerr << "   -t  Train a new model using the given script file" << std::endl;
    std::cerr << "   -e  Evaluate an existing model using the given script file" << std::endl;
    std::cerr << "   -m  Model file name. In evaluate mode this file must exists." << std::endl;
    std::cerr << "   -v  Use Viola-Jones face detector instead of computing the bounding" << std::endl;
    std::cerr << "       using the part annotations (rectangle covering all points plus" << std::endl;
    std::cerr << "       10% border)." << std::endl;
    std::cerr << "   -a  Show absolute errors. The default behavior is normalize" << std::endl;
    std::cerr << "       the error by the face size." << std::endl;
    exit(EXIT_FAILURE);
}


void main_parseOptions( int argc, char **argv )
{
    int opt;

    while ((opt = getopt(argc, argv, "t:e:m:avd:s:")) != -1)
    {
        switch (opt)
        {
            case 't':
				trainScriptFileName = string(optarg);
				break;
            case 'e':
                evaluateScriptFileName = string(optarg);
                break;
            case 'm':
				modelFileName = string(optarg);
				break;
            case 'a':
                useAbsoluteScore = true;
                break;
            case 'v':
				useViolaJones = true;
				break;
			case 's':
				configTestSplits = atoi(optarg);
				break;
			case 'd':
				configTreeDepth = atoi(optarg);
				break;
            default: /* '?' */
                main_usage();
        }
    }
    if ((trainScriptFileName.empty() && evaluateScriptFileName.empty()) || modelFileName.empty())
    {
		main_usage();
	}
}

#include <unistd.h>

int main(int argc, char** argv)
{

	ProgressIndicator pi(20);

	for (int i = 0; i < 20; ++i)
	{
		pi.update(i, true);
		sleep(3);
	}

	return 0;

	main_parseOptions(argc, argv);

	if (!trainScriptFileName.empty())
	{
		try
		{
			MainSampleLoader sloader = MainSampleLoader(useViolaJones);
			SampleList script(trainScriptFileName, &sloader);

			// create the training object
			ShapePredictorTrainer trainer;

			trainer.set_oversampling_amount(20);
			trainer.set_cascade_depth(10);
			trainer.set_num_trees_per_cascade_level(500);
			//trainer.set_nu(0.05);
			if (configTreeDepth != 0)
				trainer.set_tree_depth(configTreeDepth);
			if (configTestSplits != 0)
				trainer.set_num_test_splits(configTestSplits);
			trainer.be_verbose();

			std::cout << "      Cascade depth: " << trainer.get_cascade_depth() << std::endl;
			std::cout << "  Trees per cascade: " << trainer.get_num_trees_per_cascade_level() << std::endl;
			std::cout << "         Tree depth: " << trainer.get_tree_depth() << std::endl;
			std::cout << "Oversampling amount: " << trainer.get_oversampling_amount() << std::endl;
			std::cout << "   Number of splits: " << trainer.get_num_test_splits() << std::endl << std::endl;

			// generate the shape model and save in disk
			ShapePredictor model = trainer.train(script.getImages(), script.getAnnotations());

			if (!modelFileName.empty())
			{
				std::ofstream output(modelFileName.c_str());
				model.serialize(output);
				output.close();
			}

			cout << endl << "Mean training error: " <<
				test_shape_predictor(model, script.getImages(), script.getAnnotations(), get_interocular_distances(script.getAnnotations())) << endl;
		} catch (exception& e)
		{
			cout << "Exception thrown!" << endl;
			cout << e.what() << endl;
		}
	}
	else
	{
		try
		{
			// load the shape model from file
			ShapePredictor model;
			std::ifstream input(modelFileName.c_str());
			model.deserialize(input);
			input.close();

			MainSampleLoader sloader = MainSampleLoader(useViolaJones);
			SampleList script(evaluateScriptFileName, &sloader);

			// measures the average distance between the predicted face landmark
			// and where it should be according to the truth data.
			cout << endl << "Mean evaluating error: " <<
				test_shape_predictor(model, script.getImages(), script.getAnnotations(), get_interocular_distances(script.getAnnotations())) << endl;

		} catch (exception& e)
		{
			cout << "Exception thrown!" << endl;
			cout << e.what() << endl;
		}
    }
}
