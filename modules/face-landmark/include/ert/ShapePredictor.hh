// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SHAPE_PREDICToR_H_
#define DLIB_SHAPE_PREDICToR_H_

#include <ert/ObjectDetection.hh>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ert/RegressionTree.hh>
#include "ShapePredictorTrainer.hh"


namespace ert
{

using namespace cv;
using namespace ert;


class ShapePredictorViewer
{

	public:
		ShapePredictorViewer() {};
		virtual ~ShapePredictorViewer() {};

		virtual void show(
			int cascade,
			int tree,
			const ObjectDetection &current ) = 0;

};


class ShapePredictor : public Serializable
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
			const Rect& rect,
			ShapePredictorViewer *viewer = NULL ) const;

        unsigned long num_parts (
        ) const
        {
            return initial_shape.cols;
        }

		void serialize( std::ostream &out ) const;

		void deserialize( std::istream &in );

    private:
        Mat initial_shape;
        std::vector<std::vector<RegressionTree> > forests;
        std::vector<std::vector<unsigned long> > anchor_idx;
        std::vector<std::vector<Point2f> > deltas;
};


double test_shape_predictor (
	const ShapePredictor& sp,
	const std::vector<Mat*>& images,
	const std::vector<std::vector<ObjectDetection*> >& objects,
	const std::vector<std::vector<double> >& scales
);


}

#endif // DLIB_SHAPE_PREDICToR_H_

