
#include <opencv2/opencv.hpp>
#include "../../../modules/face-landmark/source/ert/ShapePredictorTrainer.hh"

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



void main_loadData(
	const std::string &script,
	std::vector<Mat*> &images,
	std::vector<std::vector<ObjectDetection*> > &annots  )
{
	char line[128];
	int imageCount = 0;

	ifstream list(script.c_str());

	// read the amount of files
	list.getline(line, sizeof(line));
	imageCount = atoi(line);

	images.resize(imageCount);
	annots.resize(imageCount);

	for (int i = 0; i < imageCount; ++i)
	{
		// load the image
		list.getline(line, sizeof(line) - 5);
		std::cout << "Loading " << i << " of " << imageCount << ": " << line << std::endl;
#if (0)
		cv::Mat current;
		cv::cvtColor(imread(line), current, CV_BGR2GRAY);
#else
		cv::Mat temp[3];
		split(imread(line), temp);
		temp[0].convertTo(temp[0], CV_32F);
		temp[1].convertTo(temp[1], CV_32F);
		temp[2].convertTo(temp[2], CV_32F);
		cv::Mat current = temp[0] + temp[1] + temp[2];
		current /= 3.0;
		current.convertTo(current, CV_8U);
		/*for (int r = 0; r < current.rows; ++r)
			for (int c = 0; c < current.cols; ++c)
				current.at<float>(r, c) = std::ceil( current.at<float>(r, c) );*/
		//current.convertTo(current, CV_8U);
//std::getchar();
/*std::cout << temp[0]( cv::Range(0, 5), cv::Range(0, 5) ) << std::endl
		  << temp[1]( cv::Range(0, 5), cv::Range(0, 5) ) << std::endl
		  << temp[2]( cv::Range(0, 5), cv::Range(0, 5) ) << std::endl
		  << current( cv::Range(0, 5), cv::Range(0, 5) ) << std::endl;
*/
/*cv::imshow("Test", temp[0]);
cv::waitKey(0);*/
#endif
		images[i] = new cv::Mat(current);
		// load the annotations
		int pos = strrchr(line, '.') - line;
		if (pos >= 0)
		{
			line[pos] = 0;
			strcat(line, ".pts");
		}
		else
			return;
		std::vector<ObjectDetection*> annot;
		annot.push_back( new ObjectDetection(line) );
		annots[i] = annot;
	}
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
			std::vector<Mat*> imagesTrain;
			std::vector<std::vector<ObjectDetection*> > annotsTrain;
			main_loadData(trainScriptFileName, imagesTrain, annotsTrain);

			// create the training object
			ShapePredictorTrainer trainer;

			trainer.set_oversampling_amount(20);
			trainer.set_cascade_depth(10);
			trainer.set_num_trees_per_cascade_level(500);
			trainer.set_nu(0.05);
			trainer.set_tree_depth(2);
			trainer.be_verbose();

			// generate the shape model and save in disk
			ShapePredictor sp = trainer.train(imagesTrain, annotsTrain);

			if (!modelFileName.empty())
			{
				std::ofstream output(modelFileName.c_str());
				sp.serialize(output);
				output.close();
			}

			cout << endl << "Mean training error: " <<
				test_shape_predictor(sp, imagesTrain, annotsTrain, get_interocular_distances(annotsTrain)) << endl;
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

			std::vector<Mat*> imagesEval;
			std::vector<std::vector<ObjectDetection*> > annotsEval;
			main_loadData(evaluateScriptFileName, imagesEval, annotsEval);

			// measures the average distance between the predicted face landmark
			// and where it should be according to the truth data.
			cout << endl << "Mean evaluating error: " <<
				test_shape_predictor(model, imagesEval, annotsEval, get_interocular_distances(annotsEval)) << endl;

		} catch (exception& e)
		{
			cout << "Exception thrown!" << endl;
			cout << e.what() << endl;
		}
    }
}
