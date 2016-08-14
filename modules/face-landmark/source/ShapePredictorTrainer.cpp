#include <ert/ShapePredictorTrainer.hh>
#include <ert/opencv.hh>
#include "PointAffineTransform.hh"
#include "marsene_twister.h"
#include "ProgressIndicator.hh"

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
	std::vector<Point2f>& deltas )
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
		{
			feature_pixel_values[i] = (double) img.at<uint8_t>(p.y, p.x);
//std::cout << "image(" << p.x << "," << p.y << ") = " << feature_pixel_values[i] << std::endl;
//std::getchar();
		}
		else
			feature_pixel_values[i] = 0;
//std::cout << "feature of " << p << " is " << feature_pixel_values[i] << std::endl;
	}
}



ShapePredictorTrainer::ShapePredictorTrainer ( )
{
	_cascade_depth = 10;
	_tree_depth = 4;
	_num_trees_per_cascade_level = 500;
	_nu = 0.1;
	_oversampling_amount = 20;
	_feature_pool_size = 400;
	_lambda = 0.1;
	_num_test_splits = 20;
	_feature_pool_region_padding = 0;
	_verbose = false;
	rnd = new Random();
}

unsigned long ShapePredictorTrainer::get_cascade_depth (
) const { return _cascade_depth; }

void ShapePredictorTrainer::set_cascade_depth (
	unsigned long depth
)
{
	/*DLIB_CASSERT(depth > 0,
		"\t void shape_predictor_trainer::set_cascade_depth()"
		<< "\n\t Invalid inputs were given to this function. "
		<< "\n\t depth:  " << depth
	);*/

	_cascade_depth = depth;
}

unsigned long ShapePredictorTrainer::get_tree_depth (
) const { return _tree_depth; }

void ShapePredictorTrainer::set_tree_depth (
	unsigned long depth
)
{
	/*DLIB_CASSERT(depth > 0,
		"\t void shape_predictor_trainer::set_tree_depth()"
		<< "\n\t Invalid inputs were given to this function. "
		<< "\n\t depth:  " << depth
	);*/

	_tree_depth = depth;
}

unsigned long ShapePredictorTrainer::get_num_trees_per_cascade_level (
) const { return _num_trees_per_cascade_level; }

void ShapePredictorTrainer::set_num_trees_per_cascade_level (
	unsigned long num
)
{
	/*DLIB_CASSERT( num > 0,
		"\t void shape_predictor_trainer::set_num_trees_per_cascade_level()"
		<< "\n\t Invalid inputs were given to this function. "
		<< "\n\t num:  " << num
	);*/
	_num_trees_per_cascade_level = num;
}

double ShapePredictorTrainer::get_nu (
) const { return _nu; }


void ShapePredictorTrainer::set_nu (
	double nu
)
{
	/*DLIB_CASSERT(0 < nu && nu <= 1,
		"\t void shape_predictor_trainer::set_nu()"
		<< "\n\t Invalid inputs were given to this function. "
		<< "\n\t nu:  " << nu
	);*/

	_nu = nu;
}


std::string ShapePredictorTrainer::get_random_seed (
) const { return rnd->get_seed(); }

unsigned long ShapePredictorTrainer::get_oversampling_amount (
) const { return _oversampling_amount; }


void ShapePredictorTrainer::set_oversampling_amount (
	unsigned long amount
)
{
	/*DLIB_CASSERT(amount > 0,
		"\t void shape_predictor_trainer::set_oversampling_amount()"
		<< "\n\t Invalid inputs were given to this function. "
		<< "\n\t amount: " << amount
	);*/

	_oversampling_amount = amount;
}

unsigned long ShapePredictorTrainer::get_feature_pool_size (
) const { return _feature_pool_size; }

void ShapePredictorTrainer::set_feature_pool_size (
	unsigned long size
)
{
	/*DLIB_CASSERT(size > 1,
		"\t void shape_predictor_trainer::set_feature_pool_size()"
		<< "\n\t Invalid inputs were given to this function. "
		<< "\n\t size: " << size
	);*/

	_feature_pool_size = size;
}

double ShapePredictorTrainer::get_lambda (
) const { return _lambda; }

void ShapePredictorTrainer::set_lambda (
	double lambda
)
{
	/*DLIB_CASSERT(lambda > 0,
		"\t void shape_predictor_trainer::set_lambda()"
		<< "\n\t Invalid inputs were given to this function. "
		<< "\n\t lambda: " << lambda
	);*/

	_lambda = lambda;
}

unsigned long ShapePredictorTrainer::get_num_test_splits (
) const { return _num_test_splits; }

void ShapePredictorTrainer::set_num_test_splits (
	unsigned long num
)
{
	/*DLIB_CASSERT(num > 0,
		"\t void shape_predictor_trainer::set_num_test_splits()"
		<< "\n\t Invalid inputs were given to this function. "
		<< "\n\t num: " << num
	);*/

	_num_test_splits = num;
}


double ShapePredictorTrainer::get_feature_pool_region_padding (
) const { return _feature_pool_region_padding; }


void ShapePredictorTrainer::set_feature_pool_region_padding (
	double padding
)
{
	_feature_pool_region_padding = padding;
}

void ShapePredictorTrainer::be_verbose (
)
{
	_verbose = true;
}

void ShapePredictorTrainer::be_quiet (
)
{
	_verbose = false;
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
	rnd->set_seed("HELLO");

	std::vector<TrainingSample> samples;

	// compute the initial shape guests for each training sample
	const Mat initial_shape = populate_training_sample_shapes(objects, samples);

	const std::vector<std::vector<Point2f > > pixel_coordinates = randomly_sample_pixel_coordinates(initial_shape);


	unsigned long trees_fit_so_far = 0;
	ProgressIndicator pbar(get_cascade_depth()*get_num_trees_per_cascade_level());
	if (_verbose)
		std::cout << "Fitting trees..." << std::endl;

//for (int i = 0; i < 68; ++i)
//std::cout << "part[" << i << "] = " << objects[0][0]->part(i) << std::endl;

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


/*for (int i = 0; i < pixel_coordinates[cascade].size(); ++i)
std::cout << "pixel_coordinates[" << i << "] = " << pixel_coordinates[cascade][i] << std::endl;

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
				pbar.update(trees_fit_so_far, true);
			}
		}
	}

	if (_verbose)
		std::cout << "Training complete                          " << std::endl;

	return ShapePredictor(initial_shape, forests, pixel_coordinates);
}



// CHECKED!!!
cv::Mat ShapePredictorTrainer::object_to_shape (
	const ObjectDetection& obj )
{
	// create an matrix of 2xN, where N is the amount of parts
	cv::Mat shape(2, obj.num_parts(), CV_64F);
//std::cout << "ready for tform_from_img " << obj.get_rect() << std::endl;
	const PointTransformAffine tform_from_img = normalizing_tform(obj.get_rect());
//std::cout << "tform_from_img = " << tform_from_img.get_b() << std::endl;

	for (unsigned long i = 0; i < obj.num_parts(); ++i)
	{
//std::cout << "original_point[" << i << "] = " << obj.part(i) << std::endl;
		cv::Point2f p = tform_from_img(obj.part(i));
//std::cout << "target_point[" << i << "] = " << p << std::endl;
		shape.at<double>(0, i) = p.x;
		shape.at<double>(1, i) = p.y;
	}
	return shape;
}


void ShapePredictorTrainer::printShape( const std::string& prefix, const cv::Mat& mat ) const
{
	std::cout << prefix;
	for (int i = 0; i < mat.cols; ++i)
		for (int j = 0; j < mat.rows; ++j)
			std::cout << mat.at<double>(j,i) << std::endl;
	std::cout << std::endl;
}


RegressionTree ShapePredictorTrainer::make_regression_tree (
	std::vector<TrainingSample>& samples,
	const std::vector<cv::Point2f >& pixel_coordinates
) const
{
	std::deque<std::pair<unsigned long, unsigned long> > parts;
	parts.push_back(std::make_pair(0, (unsigned long)samples.size()));

	RegressionTree tree;

	// walk the tree in breadth first order
	const unsigned long num_split_nodes = static_cast<unsigned long>(std::pow(2.0, (double)get_tree_depth())-1);

	std::vector<cv::Mat > sums(num_split_nodes*2+1);
	for (unsigned long i = 0; i < sums.size(); ++i)
		sums[i] = cv::Mat::zeros(samples[0].current_shape.rows, samples[0].current_shape.cols, samples[0].current_shape.type());

	for (unsigned long i = 0; i < samples.size(); ++i)
	{
		sums[0] += samples[i].target_shape - samples[i].current_shape;
//std::cout << "samples[" << i << "].current_shape";
//printShape(" = ", samples[i].current_shape);
	}
//printShape("sums[0] = ", sums[0]);
	for (unsigned long i = 0; i < num_split_nodes; ++i)
	{
		std::pair<unsigned long,unsigned long> range = parts.front();
		parts.pop_front();

		const SplitFeature split = generate_split(samples, range.first,
			range.second, pixel_coordinates, sums[i], sums[left_child(i)],
			sums[right_child(i)]);
		tree.splits.push_back(split);
//std::cout << "Split #" << i << " = " << split.thresh << " " << split.idx1 << " " << split.idx2  << std::endl;
		const unsigned long mid = partition_samples(split, samples, range.first, range.second);

		parts.push_back(std::make_pair(range.first, mid));
		parts.push_back(std::make_pair(mid, range.second));
	}

	// Now all the parts contain the ranges for the leaves so we can use them to
	// compute the average leaf values.
	tree.leaf_values.resize(parts.size());
	for (unsigned long i = 0; i < parts.size(); ++i)
	{
		if (parts[i].second != parts[i].first)
		{
//std::cout << parts[i].second << " != " <<  parts[i].first << std::endl;
			tree.leaf_values[i] = sums[num_split_nodes+i]*get_nu()/(parts[i].second - parts[i].first);
/*std::cout << "tree.leaf_values[" << i << "]";
printShape(" = ", tree.leaf_values[i]);*/
		}
		else
			tree.leaf_values[i] = cv::Mat::zeros(samples[0].target_shape.rows, samples[0].target_shape.cols, CV_64F);//zeros_matrix(samples[0].target_shape);

		// now adjust the current shape based on these predictions
		for (unsigned long j = parts[i].first; j < parts[i].second; ++j)
			samples[j].current_shape += tree.leaf_values[i];
	}
//std::cout << "newer samples[" << 0 << "].current_shape = " << samples[0].current_shape << std::endl;
//std::getchar();
	return tree;
}

/**
 * Create an split feature with randomly generated threshold.
 */
SplitFeature ShapePredictorTrainer::randomly_generate_split_feature (
	const std::vector<cv::Point2f >& pixel_coordinates
) const
{
	const double lambda = get_lambda();
	SplitFeature feat;
	double accept_prob;
	do
	{
		feat.idx1   = rnd->get_random_32bit_number() % get_feature_pool_size();
		feat.idx2   = rnd->get_random_32bit_number() % get_feature_pool_size();
		const double dist = length(pixel_coordinates[feat.idx1]-pixel_coordinates[feat.idx2]);
		accept_prob = std::exp(-dist/lambda);
	}
	while(feat.idx1 == feat.idx2 || !(accept_prob > rnd->get_random_double()));

	feat.thresh = (rnd->get_random_double()*256 - 128)/2.0;

	return feat;
}


/**
 * Generate a bunch of random splits, test them and return the best one.
 */
SplitFeature ShapePredictorTrainer::generate_split (
	const std::vector<TrainingSample>& samples,
	unsigned long begin,
	unsigned long end,
	const std::vector<cv::Point2f >& pixel_coordinates,
	const cv::Mat& sum,
	cv::Mat& left_sum,
	cv::Mat& right_sum
) const
{
	const unsigned long num_test_splits = get_num_test_splits();

	// sample the random features we test in this function
	std::vector<SplitFeature> feats;
	feats.reserve(num_test_splits);
	for (unsigned long i = 0; i < num_test_splits; ++i)
		feats.push_back(randomly_generate_split_feature(pixel_coordinates));

	std::vector<cv::Mat > left_sums(num_test_splits);
	std::vector<unsigned long> left_cnt(num_test_splits);

	// now compute the sums of vectors that go left for each feature
	Mat temp;
	for (unsigned long j = begin; j < end; ++j)
	{
		temp = samples[j].target_shape-samples[j].current_shape;
		for (unsigned long i = 0; i < num_test_splits; ++i)
		{
			if (samples[j].feature_pixel_values[feats[i].idx1] - samples[j].feature_pixel_values[feats[i].idx2] > feats[i].thresh)
			{
				if (left_sums[i].rows == 0)
					temp.copyTo(left_sums[i]);
				else
					left_sums[i] += temp;
				++left_cnt[i];
			}
		}
	}

	// now figure out which feature is the best
	double best_score = -1;
	unsigned long best_feat = 0;
	for (unsigned long i = 0; i < num_test_splits; ++i)
	{
		// check how well the feature splits the space.
		double score = 0;
		unsigned long right_cnt = end-begin-left_cnt[i];
		if (left_cnt[i] != 0 && right_cnt != 0)
		{
			temp = sum - left_sums[i];
			score = left_sums[i].dot(left_sums[i])/left_cnt[i] + temp.dot(temp)/right_cnt;
			if (score > best_score)
			{
				best_score = score;
				best_feat = i;
			}
		}
	}

	cv::swap(left_sums[best_feat], left_sum);
	//if (left_sum.size() != 0)
	if (left_sum.rows != 0)
	{
		right_sum = cv::Mat(sum - left_sum);
	}
	else
	{
		//right_sum = sum;
		sum.copyTo(right_sum);
		left_sum = cv::Mat::zeros(sum.rows, sum.cols, sum.type());
	}
	return feats[best_feat];
	//return impl::SplitFeature();
}

/**
 * Splits samples based on split (sorta like in quick sort) and returns the mid
 * point.  make sure you return the mid in a way compatible with how we walk
 * through the tree.
 */
unsigned long ShapePredictorTrainer::partition_samples (
	const SplitFeature& split,
	std::vector<TrainingSample>& samples,
	unsigned long begin,
	unsigned long end
) const
{
	//
//std::cout << "range = " << begin << " to " << end << std::endl;
	unsigned long i = begin;
	for (unsigned long j = begin; j < end; ++j)
	{
//std::cout << samples[j].feature_pixel_values[split.idx1] << " - " << samples[j].feature_pixel_values[split.idx2] << "> " << split.thresh << std::endl;
		if (samples[j].feature_pixel_values[split.idx1] - samples[j].feature_pixel_values[split.idx2] > split.thresh)
		{
			samples[i].swap(samples[j]);
			++i;
		}
	}

	return i;
}



cv::Mat ShapePredictorTrainer::populate_training_sample_shapes(
	const std::vector<std::vector<ObjectDetection*> >& objects,
	std::vector<TrainingSample>& samples
) const
{
	samples.clear();
	cv::Mat mean_shape;
	long count = 0;

	// first fill out the target shapes
	for (unsigned long i = 0; i < objects.size(); ++i)
	{
		for (unsigned long j = 0; j < objects[i].size(); ++j)
		{
			TrainingSample sample;
			sample.image_idx = i;
			sample.rect = objects[i][j]->get_rect();
			sample.target_shape = object_to_shape(*objects[i][j]);

			for (unsigned long itr = 0; itr < get_oversampling_amount(); ++itr)
				samples.push_back(sample);

			// sum the current shape to the mean shape
			if (mean_shape.rows == 0)
				sample.target_shape.copyTo(mean_shape);
			else
				mean_shape += sample.target_shape;
			++count;
		}
	}

	// compute the mean shape
	mean_shape /= count;

	// now go pick random initial shapes
	for (unsigned long i = 0; i < samples.size(); ++i)
	{
		if (false || (i%get_oversampling_amount()) == 0)
		{
			// The mean shape is what we really use as an initial shape so always
			// include it in the training set as an example starting shape.
			mean_shape.copyTo(samples[i].current_shape);
		}
		else
		{
			// Pick a random convex combination of two of the target shapes and use
			// that as the initial shape for this sample.
			const unsigned long rand_idx = rnd->get_random_32bit_number() % samples.size();
			const unsigned long rand_idx2 = rnd->get_random_32bit_number() % samples.size();
			const double alpha = rnd->get_random_double();
			samples[i].current_shape = alpha*samples[rand_idx].target_shape + (1-alpha)*samples[rand_idx2].target_shape;
//std::cout << mean_shape << std::endl;
//std::cout << samples[i].current_shape << std::endl;
//std::getchar();
		}
	}

//std::cout << mean_shape << std::endl;
//std::getchar();

	return mean_shape;
}


void ShapePredictorTrainer::randomly_sample_pixel_coordinates (
	std::vector<cv::Point2f>& pixel_coordinates,
	const double min_x,
	const double min_y,
	const double max_x,
	const double max_y
) const
/*!
	ensures
		- #pixel_coordinates.size() == get_feature_pool_size()
		- for all valid i:
			- pixel_coordinates[i] == a point in the box defined by the min/max x/y arguments.
!*/
{
	pixel_coordinates.resize(get_feature_pool_size());
	for (unsigned long i = 0; i < get_feature_pool_size(); ++i)
	{
		pixel_coordinates[i].x = rnd->get_random_double()*(max_x-min_x) + min_x;
		pixel_coordinates[i].y = rnd->get_random_double()*(max_y-min_y) + min_y;
//std::cout << "pixel_coordinates[" << i << "] = " << pixel_coordinates[i] << std::endl;
//std::getchar();
	}
}

std::vector<std::vector<cv::Point2f > > ShapePredictorTrainer::randomly_sample_pixel_coordinates (
	const cv::Mat& initial_shape
) const
{
	const double padding = get_feature_pool_region_padding();
	// Figure figure out the bounds on the object shapes.  We will sample uniformly
	// from this box.

	double min_x, max_x;
	cv::minMaxLoc( initial_shape( cv::Range(0, 1), cv::Range::all() ), &min_x, &max_x );
	min_x -= padding;
	max_x -= padding;

	double min_y, max_y;
	cv::minMaxLoc( initial_shape( cv::Range(1, 2), cv::Range::all() ), &min_y, &max_y );
	min_y += padding;
	max_y += padding;

//std::cout << "randomly_sample_pixel_coordinates: " << min_x << "  " << min_y << "  " << max_y << "  " << max_y << std::endl;
//std::getchar();

	std::vector<std::vector<cv::Point2f > > pixel_coordinates;
	pixel_coordinates.resize(get_cascade_depth());
	for (unsigned long i = 0; i < get_cascade_depth(); ++i)
		randomly_sample_pixel_coordinates(pixel_coordinates[i], min_x, min_y, max_x, max_y);
	return pixel_coordinates;
}


};
