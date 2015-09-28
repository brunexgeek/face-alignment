#include "ShapePredictorTrainer.hh"
#include "opencv.hh"


namespace ert{




	Point2f location (
		const Mat& shape,
		unsigned long idx
	)
	/*!
		requires
			- idx < shape.size()/2
			- shape.size()%2 == 0
		ensures
			- returns the idx-th point from the shape vector.
	!*/
	{
		return Point2f(shape.at<double>(0,idx), shape.at<double>(1,idx));
	}



	double length( const Point2f &p )
	{
		return std::sqrt( (p.x * p.x) + (p.y * p.y) );
		//return (double)p.x + (double)p.y;
	}


	/**
	 * Find the nearest part of the shape to this pixel
	 */
	inline unsigned long nearest_shape_point (
		const Mat& shape,
		const Point2f& pt
	)
	{
		float best_dist = std::numeric_limits<float>::infinity();
		const unsigned long num_shape_parts = shape.cols;
		unsigned long best_idx = 0;
		for (unsigned long j = 0; j < num_shape_parts; ++j)
		{
			const float dist = length_squared(location(shape,j)-pt);
//std::cout << "dist from " << location(shape,j) << " to " << pt << " = " << dist;
			if (dist < best_dist)
			{
				best_dist = dist;
				best_idx = j;
//std::cout << "*";
			}
//std::cout << std::endl;
		}
		return best_idx;
	}




/**
* Given an array of image pixel coordinates, computes the delta between that
* coordinate to the nearest part in the shape.
*/
void create_shape_relative_encoding (
const Mat& shape,
const std::vector<Point2f> &pixel_coordinates,
std::vector<unsigned long>& anchor_idx,
std::vector<Point2f>& deltas
)
/*!
requires
	- shape.size()%2 == 0
	- shape.size() > 0
ensures
	- #anchor_idx.size() == pixel_coordinates.size()
	- #deltas.size()     == pixel_coordinates.size()
	- for all valid i:
		- pixel_coordinates[i] == location(shape,#anchor_idx[i]) + #deltas[i]
!*/
{
anchor_idx.resize(pixel_coordinates.size());
deltas.resize(pixel_coordinates.size());


for (unsigned long i = 0; i < pixel_coordinates.size(); ++i)
{
	anchor_idx[i] = nearest_shape_point(shape, pixel_coordinates[i]);
	deltas[i] = pixel_coordinates[i] - location(shape,anchor_idx[i]);
}
}

// ------------------------------------------------------------------------------------

inline PointTransformAffine find_tform_between_shapes (
	const Mat& from_shape,
	const Mat& to_shape
)
{
	//DLIB_ASSERT(from_shape.size() == to_shape.size() && (from_shape.size()%2) == 0 && from_shape.size() > 0,"");
	std::vector<Point2f> from_points, to_points;
	const unsigned long num = from_shape.cols;
	from_points.reserve(num);
	to_points.reserve(num);
	if (num == 1)
	{
		// Just use an identity transform if there is only one landmark.
		return PointTransformAffine();
	}

	for (unsigned long i = 0; i < num; ++i)
	{
		from_points.push_back(location(from_shape,i));
		to_points.push_back(location(to_shape,i));
	}
	return PointTransformAffine::findSimilarityTransform(from_points, to_points);
}

// ------------------------------------------------------------------------------------

// CHECKED!!!
PointTransformAffine normalizing_tform (
	const Rect& rect
)
/*!
	ensures
		- returns a transform that maps rect.tl_corner() to (0,0) and rect.br_corner()
		  to (1,1).
!*/
{
	std::vector<Point2f> from_points, to_points;
	from_points.push_back( Point2f(rect.x, rect.y) );
	to_points.push_back(Point2f(0,0));
	from_points.push_back( Point2f(rect.x + rect.width, rect.y) );
	to_points.push_back(Point2f(1,0));
	from_points.push_back( Point2f(rect.x + rect.width, rect.y + rect.height) );
	to_points.push_back(Point2f(1,1));
	return PointTransformAffine::findAffineTransform(from_points, to_points);
}

// ------------------------------------------------------------------------------------

PointTransformAffine unnormalizing_tform (
	const Rect& rect
)
/*!
	ensures
		- returns a transform that maps (0,0) to rect.tl_corner() and (1,1) to
		  rect.br_corner().
!*/
{
	std::vector<Point2f> from_points, to_points;
	to_points.push_back( Point2f(rect.x, rect.y) );
	from_points.push_back(Point(0,0));
	to_points.push_back( Point2f(rect.x + rect.width, rect.y) );
	from_points.push_back(Point(1,0));
	to_points.push_back(Point2f(rect.x + rect.width, rect.y + rect.height) );
	from_points.push_back(Point(1,1));
	return PointTransformAffine::findAffineTransform(from_points, to_points);
}

// ------------------------------------------------------------------------------------

void extract_feature_pixel_values (
	const Mat& img,
	const Rect& rect,
	const Mat& current_shape,
	const Mat& reference_shape,
	const std::vector<unsigned long>& reference_pixel_anchor_idx,
	const std::vector<Point2f>& reference_pixel_deltas,
	std::vector<double>& feature_pixel_values
)
/*!
	requires
		- image_type == an image object that implements the interface defined in
		  dlib/image_processing/generic_image.h
		- reference_pixel_anchor_idx.size() == reference_pixel_deltas.size()
		- current_shape.size() == reference_shape.size()
		- reference_shape.size()%2 == 0
		- max(mat(reference_pixel_anchor_idx)) < reference_shape.size()/2
	ensures
		- #feature_pixel_values.size() == reference_pixel_deltas.size()
		- for all valid i:
			- #feature_pixel_values[i] == the value of the pixel in img_ that
			  corresponds to the pixel identified by reference_pixel_anchor_idx[i]
			  and reference_pixel_deltas[i] when the pixel is located relative to
			  current_shape rather than reference_shape.
!*/
{
	assert(img.type() == CV_8UC1);
	const Mat tform = find_tform_between_shapes(reference_shape, current_shape).get_m();
//std::cout << "tform = \n" << tform << std::endl;
//std::getchar();
	const PointTransformAffine tform_to_img = unnormalizing_tform(rect);

	const Rect area = Rect(0, 0, img.cols, img.rows);

	//const_image_view<image_type> img(img_);
	feature_pixel_values.resize(reference_pixel_deltas.size());
	for (unsigned long i = 0; i < feature_pixel_values.size(); ++i)
	{
		// Compute the Point in the current shape corresponding to the i-th pixel and
		// then map it from the normalized shape space into pixel space.
		Point2f p = tform_to_img(tform*reference_pixel_deltas[i] + location(current_shape, reference_pixel_anchor_idx[i]));
		p.x = round(p.x);
		p.y = round(p.y);
		if (area.contains(p))
			feature_pixel_values[i] = (double) img.at<uint8_t>(p.y, p.x);
		else
			feature_pixel_values[i] = 0;
//std::cout << "feature of " << p << " is " << feature_pixel_values[i] << std::endl;
	}
}



ShapePredictor ShapePredictorTrainer::train (
	const std::vector<Mat*>& images,
	const std::vector<std::vector<ObjectDetection*> >& objects
) const
{
	assert(images.size() == objects.size() && images.size() > 0);
	/*DLIB_CASSERT(images.size() == objects.size() && images.size() > 0,
		"\t shape_predictor shape_predictor_trainer::train()"
		<< "\n\t Invalid inputs were given to this function. "
		<< "\n\t images.size():  " << images.size()
		<< "\n\t objects.size(): " << objects.size()
	);*/

	// make sure the objects agree on the number of parts and that there is at
	// least one FullObjectDetection.
	unsigned long num_parts = 0;
	for (unsigned long i = 0; i < objects.size(); ++i)
	{
		for (unsigned long j = 0; j < objects[i].size(); ++j)
		{
			if (num_parts == 0)
			{
				num_parts = objects[i][j]->num_parts();
				assert(objects[i][j]->num_parts() != 0);
				/*DLIB_CASSERT(objects[i][j].num_parts() != 0,
					"\t shape_predictor shape_predictor_trainer::train()"
					<< "\n\t You can't give objects that don't have any parts to the trainer."
				);*/
			}
			else
			{
				assert(objects[i][j]->num_parts() == num_parts);
				/*DLIB_CASSERT(objects[i][j].num_parts() == num_parts,
					"\t shape_predictor shape_predictor_trainer::train()"
					<< "\n\t All the objects must agree on the number of parts. "
					<< "\n\t objects["<<i<<"]["<<j<<"].num_parts(): " << objects[i][j].num_parts()
					<< "\n\t num_parts:  " << num_parts
				);*/
			}
		}
	}
	/*DLIB_CASSERT(num_parts != 0,
		"\t shape_predictor shape_predictor_trainer::train()"
		<< "\n\t You must give at least one full_object_detection if you want to train a shape model and it must have parts."
	);*/

	//rnd.set_seed(get_random_seed());
	rnd.set_seed("HELLO");

	std::vector<TrainingSample> samples;

	// compute the initial shape guests for each training sample
	const Mat initial_shape = populate_training_sample_shapes(objects, samples);

	const std::vector<std::vector<Point2f > > pixel_coordinates = randomly_sample_pixel_coordinates(initial_shape);


	unsigned long trees_fit_so_far = 0;
	console_progress_indicator pbar(get_cascade_depth()*get_num_trees_per_cascade_level());
	if (_verbose)
		std::cout << "Fitting trees..." << std::endl;

	std::vector<std::vector<RegressionTree> > forests(get_cascade_depth());

	// Now start doing the actual training by filling in the forests
	for (unsigned long cascade = 0; cascade < get_cascade_depth(); ++cascade)
	{
//std::cout << "initial_shape = " << std::endl << initial_shape << std::endl << std::endl;
		// Each cascade uses a different set of pixels for its features.  We compute
		// their representations relative to the initial shape first.
		std::vector<unsigned long> anchor_idx;
		std::vector<Point2f > deltas;
		create_shape_relative_encoding(initial_shape, pixel_coordinates[cascade], anchor_idx, deltas);


/*for (int i = 0; i < anchor_idx.size(); ++i)
std::cout << "anchor_idx[" << i << "] = " << anchor_idx[i] << std::endl;

std::getchar();*/


		// First compute the feature_pixel_values for each training sample at this
		// level of the cascade.
		for (unsigned long i = 0; i < samples.size(); ++i)
		{
			extract_feature_pixel_values(*images[samples[i].image_idx], samples[i].rect,
				samples[i].current_shape, initial_shape, anchor_idx,
				deltas, samples[i].feature_pixel_values);
		}

		// Now start building the trees at this cascade level.
		forests[cascade].reserve( get_num_trees_per_cascade_level() );
		for (unsigned long i = 0; i < get_num_trees_per_cascade_level(); ++i)
		{
			forests[cascade].push_back(make_regression_tree(samples, pixel_coordinates[cascade]));

			if (_verbose)
			{
				++trees_fit_so_far;
				pbar.print_status(trees_fit_so_far);
			}
		}
	}

	if (_verbose)
		std::cout << "Training complete                          " << std::endl;

	return ShapePredictor(initial_shape, forests, pixel_coordinates);
}


};
