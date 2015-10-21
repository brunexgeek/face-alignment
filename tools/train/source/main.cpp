
#include <opencv2/opencv.hpp>
#include "../../../modules/face-landmark/source/ert/ShapePredictorTrainer.hh"
#include "../../../modules/face-landmark/source/ert/SampleList.hh"

#include <iostream>
#include <fstream>
#include <cstring>
#include <getopt.h>

using namespace ert;
using namespace std;


string trainScriptFileName = "";

string evaluateScriptFileName = "";

string modelFileName = "";

bool absoluteScore = false;


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
    std::cerr << "Usage: tool_train -t <script file> -m <model file> [ -a ]" << std::endl;
    std::cerr << "       tool_train -e <script file> -m <model file> [ -a ]" << std::endl;
    exit(EXIT_FAILURE);
}


void main_parseOptions( int argc, char **argv )
{
    int opt;

    while ((opt = getopt(argc, argv, "t:e:m:a")) != -1)
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
                absoluteScore = true;
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

int main(int argc, char** argv)
{
	main_parseOptions(argc, argv);

	if (!trainScriptFileName.empty())
	{
		try
		{
			SampleList script(trainScriptFileName);

			// create the training object
			ShapePredictorTrainer trainer;

			trainer.set_oversampling_amount(20);
			trainer.set_cascade_depth(10);
			trainer.set_num_trees_per_cascade_level(500);
			trainer.set_nu(0.05);
			trainer.set_tree_depth(2);
			trainer.be_verbose();

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

			SampleList script(evaluateScriptFileName);

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
