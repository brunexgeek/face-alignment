
#include <opencv2/opencv.hpp>
#include <face-detector/detector.hpp>
#include <ert/ShapePredictorTrainer.hh>
#include <ert/SampleList.hh>

#include <iostream>
#include <fstream>
#include <cstring>
#include <getopt.h>


#define FACE_CASCADE_FILE     "haarcascade_frontalface_alt.xml"


using namespace ert;
using namespace std;
using namespace vasr::detector;


string evaluateScriptFileName = "";

string modelFileName = "";


void main_usage()
{
    std::cerr << "Usage: tool_test -e <script file> -m <model file>" << std::endl;
    exit(EXIT_FAILURE);
}


void main_parseOptions( int argc, char **argv )
{
    int opt;

    while ((opt = getopt(argc, argv, "e:m:")) != -1)
    {
        switch (opt)
        {
            case 'e':
                evaluateScriptFileName = string(optarg);
                break;
            case 'm':
				modelFileName = string(optarg);
				break;
            default: /* '?' */
                main_usage();
        }
    }
    if (evaluateScriptFileName.empty() || modelFileName.empty())
    {
		main_usage();
	}
}


double main_length( const Point2f &p )
{
	return std::sqrt( (p.x * p.x) + (p.y * p.y) );
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

		SampleList script(evaluateScriptFileName);

		CascadeClassifier faceCascade(FACE_CASCADE_FILE);
		ViolaJones *detector = new ViolaJones(&faceCascade);

		for (size_t i = 0; i < script.getImages().size(); ++i)
        {
			// using only the first face in each image

			Rect face;
#if (0)
			if (!detector->detect(*script.getImages()[i], face)) continue;

			Rect diff = script.getAnnotations()[i][0]->get_rect();
			diff.x -= face.x;
			diff.y -= face.y;
			diff.width -= face.width;
			diff.height -= face.height;
			std::cout << "Original - Viola-Jones = " << diff << std::endl;
#endif
			face = script.getAnnotations()[i][0]->get_rect();

			ObjectDetection det = model.detect(*script.getImages()[i], face);
			std::string fitFileName = script.getFileName(i, "fit");
			std::cout << "Saving fitted points to " <<  fitFileName << std::endl;
			det.save(fitFileName);
#if (0)
			for (int j = 0; j < (int)det.num_parts(); ++j)
			{
					std::cout << "Gold: " << script.getAnnotations()[i][0]->part(j) <<
						"\tFit: " << det.part(j) << std::endl;
			}
#endif
        }

        delete detector;

	} catch (exception& e)
	{
		cout << "Exception thrown!" << endl;
		cout << e.what() << endl;
	}
}
