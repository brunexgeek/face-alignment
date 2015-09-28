// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SHAPE_PREDICToR_H_
#define DLIB_SHAPE_PREDICToR_H_

#include "ObjectDetection.hh"
#include <opencv2/opencv.hpp>
#include "PointAffineTransform.hh"
#include "console_progress_indicator.h"
#include "Serializable.hh"
#include <iostream>
#include "rand/rand_kernel_1.h"
#include "RegressionTree.hh"
#include "ShapePredictorTrainer.hh"


namespace ert
{

	using namespace cv;
	using namespace ert;


/*
cv::RNG& getRnd()
{
	static cv::RNG rnd( clock() );

	return rnd;
}


double get_random_double (
)
{
	uint32_t temp;

	double max_val =  0xFFFFFF;
	max_val *= 0x1000000;
	max_val += 0xFFFFFF;
	max_val += 0.01;


	temp = getRnd().uniform(0, 0x00FFFFFF);
	temp &= 0xFFFFFF;

	double val = static_cast<double>(temp);

	val *= 0x1000000;

	temp = getRnd().uniform(0, 0x00FFFFFF);
	temp &= 0xFFFFFF;

	val += temp;

	val /= max_val;

	if (val < 1.0)
	{
		return val;
	}
	else
	{
		// return a value slightly less than 1.0
		return 1.0 - std::numeric_limits<double>::epsilon();
	}
}
*/





// ----------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------



class ShapePredictor
{
	public:

        ShapePredictor ()
        {
			// nothing to do
		}

        ShapePredictor (
            const Mat& initial_shape_,
            const std::vector<std::vector<RegressionTree> >& forests_,
            const std::vector<std::vector<Point2f > >& pixel_coordinates
        );

		ObjectDetection detect(
			const Mat& img,
			const Rect& rect
		) const;

        unsigned long num_parts (
        ) const
        {
            return initial_shape.cols / 2;
        }

    private:
        Mat initial_shape;
        std::vector<std::vector<RegressionTree> > forests;
        std::vector<std::vector<unsigned long> > anchor_idx;
        std::vector<std::vector<Point2f> > deltas;
};

// ----------------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------

   /* template <
        typename image_array
        >
    double test_shape_predictor (
        const ShapePredictor& sp,
        const std::vector<Mat*>& images,
        const std::vector<std::vector<ObjectDetection*> >& objects
    )
    {
        std::vector<std::vector<double> > no_scales;
        return test_shape_predictor(sp, images, objects, no_scales);
    }*/


    double test_shape_predictor (
        const ShapePredictor& sp,
        const std::vector<Mat*>& images,
        const std::vector<std::vector<ObjectDetection*> >& objects,
        const std::vector<std::vector<double> >& scales
    );


// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SHAPE_PREDICToR_H_

