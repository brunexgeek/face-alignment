// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SHAPE_PREDICToR_H_
#define DLIB_SHAPE_PREDICToR_H_

#include "shape_predictor_abstract.h"
#include "full_object_detection.h"
#include <opencv2/opencv.hpp>
#include "point_transforms.h"
#include "console_progress_indicator.h"
#include "serialize.h"
#include "Serializable.hh"
#include <iostream>

namespace dlib
{

using namespace cv;


void printRow( std::ostream& os, bool isX, const FullObjectDetection& obj )
{
	int c;
	bool f;

	if (isX)
		os << "X = [ ";
	else
		os << "Y = [ ";

	for (c = 0, f = false; c < obj.num_parts(); ++c)
	{
		//if (obj.num_parts() > 20)
		{
			//if ((c < 10) || (c > (obj.num_parts() - 10)))
				if (isX)
					os << round(obj.part(c).x) << " ";
				else
					os << round(obj.part(c).y) << " ";
			/*else
			{
				if (!f)
				{
					std::cout << " ... ";
					f = true;
				}
			}*/
		}
	}
	os << "]" << std::endl;
}


std::ostream& operator<<(std::ostream& os, const FullObjectDetection& obj)
{
	printRow(os, true, obj);
	printRow(os, false, obj);
}


cv::RNG& getRnd()
{
	static cv::RNG rnd( clock() );

	return rnd;
}


// ----------------------------------------------------------------------------------------

    namespace impl
    {
        struct SplitFeature : public ert::Serializable
        {
            unsigned long idx1;
            unsigned long idx2;
            float thresh;

            void serialize ( std::ostream& out)
            {
                /*dlib::serialize(item.idx1, out);
                dlib::serialize(item.idx2, out);
                dlib::serialize(item.thresh, out);*/
            }
            void deserialize ( std::istream& in)
            {
                /*dlib::deserialize(item.idx1, in);
                dlib::deserialize(item.idx2, in);
                dlib::deserialize(item.thresh, in);*/
            }
        };


        // a tree is just a std::vector<impl::SplitFeature>.  We use this function to navigate the
        // tree nodes
        inline unsigned long left_child (unsigned long idx) { return 2*idx + 1; }
        /*!
            ensures
                - returns the index of the left child of the binary tree node idx
        !*/
        inline unsigned long right_child (unsigned long idx) { return 2*idx + 2; }
        /*!
            ensures
                - returns the index of the left child of the binary tree node idx
        !*/

        struct RegressionTree
        {
            std::vector<SplitFeature> splits;
            std::vector<Mat> leaf_values;

            inline const Mat& operator()(
                const std::vector<float>& feature_pixel_values
            ) const
            /*!
                requires
                    - All the index values in splits are less than feature_pixel_values.size()
                    - leaf_values.size() is a power of 2.
                      (i.e. we require a tree with all the levels fully filled out.
                    - leaf_values.size() == splits.size()+1
                      (i.e. there needs to be the right number of leaves given the number of splits in the tree)
                ensures
                    - runs through the tree and returns the vector at the leaf we end up in.
            !*/
            {
                unsigned long i = 0;
                while (i < splits.size())
                {
                    if (feature_pixel_values[splits[i].idx1] - feature_pixel_values[splits[i].idx2] > splits[i].thresh)
                        i = left_child(i);
                    else
                        i = right_child(i);
                }
                return leaf_values[i - splits.size()];
            }

            friend void serialize (const RegressionTree& item, std::ostream& out)
            {
                /*dlib::serialize(item.splits, out);
                dlib::serialize(item.leaf_values, out);*/
            }
            friend void deserialize (RegressionTree& item, std::istream& in)
            {
                /*dlib::deserialize(item.splits, in);
                dlib::deserialize(item.leaf_values, in);*/
            }
        };

    // ------------------------------------------------------------------------------------

        inline Point2f location (
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

    // ------------------------------------------------------------------------------------

	double length_squared( const Point2f &p )
	{
		return (double)p.x * (double)p.x + (double)p.y * (double)p.y;
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
		const unsigned long num_shape_parts = shape.cols / 2;
		unsigned long best_idx = 0;
		for (unsigned long j = 0; j < num_shape_parts; ++j)
		{
			const float dist = length_squared(location(shape,j)-pt);
			if (dist < best_dist)
			{
				best_dist = dist;
				best_idx = j;
			}
		}
		return best_idx;
	}

    // ------------------------------------------------------------------------------------

	/**
	 * Given an array of image pixel coordinates, computes the delta between that
	 * coordinate to the nearest part in the shape.
	 */
	inline void create_shape_relative_encoding (
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

        inline point_transform_affine find_tform_between_shapes (
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
                return point_transform_affine();
            }

            for (unsigned long i = 0; i < num; ++i)
            {
                from_points.push_back(location(from_shape,i));
                to_points.push_back(location(to_shape,i));
            }
            return find_similarity_transform(from_points, to_points);
        }

    // ------------------------------------------------------------------------------------

        inline point_transform_affine normalizing_tform (
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
            return find_affine_transform(from_points, to_points);
        }

    // ------------------------------------------------------------------------------------

        inline point_transform_affine unnormalizing_tform (
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
            return find_affine_transform(from_points, to_points);
        }

    // ------------------------------------------------------------------------------------

        void extract_feature_pixel_values (
            const Mat& img,
            const Rect& rect,
            const Mat& current_shape,
            const Mat& reference_shape,
            const std::vector<unsigned long>& reference_pixel_anchor_idx,
            const std::vector<Point2f>& reference_pixel_deltas,
            std::vector<float>& feature_pixel_values
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
			//assert(img.type() == CV_32F);
            const Mat tform = find_tform_between_shapes(reference_shape, current_shape).get_m();
            const point_transform_affine tform_to_img = unnormalizing_tform(rect);

            const Rect area = Rect(0, 0, img.cols, img.rows);

            //const_image_view<image_type> img(img_);
            feature_pixel_values.resize(reference_pixel_deltas.size());
            for (unsigned long i = 0; i < feature_pixel_values.size(); ++i)
            {
                // Compute the Point in the current shape corresponding to the i-th pixel and
                // then map it from the normalized shape space into pixel space.
                Point p = tform_to_img(tform*reference_pixel_deltas[i] + location(current_shape, reference_pixel_anchor_idx[i]));
                if (area.contains(p))
                    feature_pixel_values[i] = img.at<float>(p.y, p.x);
                else
                    feature_pixel_values[i] = 0;
            }
        }

    } // end namespace impl

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
            const std::vector<std::vector<impl::RegressionTree> >& forests_,
            const std::vector<std::vector<Point2f > >& pixel_coordinates
        ) : initial_shape(initial_shape_), forests(forests_)
        /*!
            requires
                - initial_shape.size()%2 == 0
                - forests.size() == pixel_coordinates.size() == the number of cascades
                - for all valid i:
                    - all the index values in forests[i] are less than pixel_coordinates[i].size()
                - for all valid i and j:
                    - forests[i][j].leaf_values.size() is a power of 2.
                      (i.e. we require a tree with all the levels fully filled out.
                    - forests[i][j].leaf_values.size() == forests[i][j].splits.size()+1
                      (i.e. there need to be the right number of leaves given the number of splits in the tree)
        !*/
        {
            anchor_idx.resize(pixel_coordinates.size());
            deltas.resize(pixel_coordinates.size());
            // Each cascade uses a different set of pixels for its features.  We compute
            // their representations relative to the initial shape now and save it.
            for (unsigned long i = 0; i < pixel_coordinates.size(); ++i)
                impl::create_shape_relative_encoding(initial_shape, pixel_coordinates[i], anchor_idx[i], deltas[i]);
        }

        unsigned long num_parts (
        ) const
        {
            return initial_shape.cols / 2;
        }

        FullObjectDetection detect(
            const Mat& img,
            const Rect& rect
        ) const
        {
            using namespace impl;
            Mat current_shape = initial_shape;
            std::vector<float> feature_pixel_values;
            for (unsigned long iter = 0; iter < forests.size(); ++iter)
            {
                extract_feature_pixel_values(img, rect, current_shape, initial_shape, anchor_idx[iter], deltas[iter], feature_pixel_values);
                // evaluate all the trees at this level of the cascade.
                for (unsigned long i = 0; i < forests[iter].size(); ++i)
                    current_shape += forests[iter][i](feature_pixel_values);
            }

            // convert the current_shape into a full_object_detection
            const point_transform_affine tform_to_img = unnormalizing_tform(rect);
            std::vector<Point2f> parts(current_shape.cols);
            for (unsigned long i = 0; i < parts.size(); ++i)
                parts[i] = tform_to_img(location(current_shape, i));
            return FullObjectDetection(rect, parts);
        }

        friend void serialize (const ShapePredictor& item, std::ostream& out)
        {
            /*int version = 1;
            dlib::serialize(version, out);
            dlib::serialize(item.initial_shape, out);
            dlib::serialize(item.forests, out);
            dlib::serialize(item.anchor_idx, out);
            dlib::serialize(item.deltas, out);*/
        }
        friend void deserialize (ShapePredictor& item, std::istream& in)
        {
            /*int version = 0;
            dlib::deserialize(version, in);
            if (version != 1)
                throw serialization_error("Unexpected version found while deserializing dlib::ShapePredictor.");
            dlib::deserialize(item.initial_shape, in);
            dlib::deserialize(item.forests, in);
            dlib::deserialize(item.anchor_idx, in);
            dlib::deserialize(item.deltas, in);*/
        }

    private:
        Mat initial_shape;
        std::vector<std::vector<impl::RegressionTree> > forests;
        std::vector<std::vector<unsigned long> > anchor_idx;
        std::vector<std::vector<Point2f> > deltas;
};

// ----------------------------------------------------------------------------------------

class ShapePredictorTrainer
{
        /*!
            This thing really only works with unsigned char or rgb_pixel images (since we assume the threshold
            should be in the range [-128,128]).
        !*/
    public:

        ShapePredictorTrainer ( )
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
        }

        unsigned long get_cascade_depth (
        ) const { return _cascade_depth; }

        void set_cascade_depth (
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

        unsigned long get_tree_depth (
        ) const { return _tree_depth; }

        void set_tree_depth (
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

        unsigned long get_num_trees_per_cascade_level (
        ) const { return _num_trees_per_cascade_level; }

        void set_num_trees_per_cascade_level (
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

        double get_nu (
        ) const { return _nu; }
        void set_nu (
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

        unsigned long get_oversampling_amount (
        ) const { return _oversampling_amount; }
        void set_oversampling_amount (
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

        unsigned long get_feature_pool_size (
        ) const { return _feature_pool_size; }
        void set_feature_pool_size (
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

        double get_lambda (
        ) const { return _lambda; }
        void set_lambda (
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

        unsigned long get_num_test_splits (
        ) const { return _num_test_splits; }
        void set_num_test_splits (
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


        double get_feature_pool_region_padding (
        ) const { return _feature_pool_region_padding; }
        void set_feature_pool_region_padding (
            double padding
        )
        {
            _feature_pool_region_padding = padding;
        }

        void be_verbose (
        )
        {
            _verbose = true;
        }

        void be_quiet (
        )
        {
            _verbose = false;
        }

        ShapePredictor train (
            const std::vector<Mat*>& images,
            const std::vector<std::vector<FullObjectDetection*> >& objects
        ) const
        {
            using namespace impl;
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

            std::vector<TrainingSample> samples;

            // compute the initial shape guests for each training sample
            const Mat initial_shape = populate_training_sample_shapes(objects, samples);


            const std::vector<std::vector<Point2f > > pixel_coordinates = randomly_sample_pixel_coordinates(initial_shape);

            unsigned long trees_fit_so_far = 0;
            console_progress_indicator pbar(get_cascade_depth()*get_num_trees_per_cascade_level());
            if (_verbose)
                std::cout << "Fitting trees..." << std::endl;

            std::vector<std::vector<impl::RegressionTree> > forests(get_cascade_depth());

            // Now start doing the actual training by filling in the forests
            for (unsigned long cascade = 0; cascade < get_cascade_depth(); ++cascade)
            {
std::cout << "initial_shape = " << std::endl << initial_shape << std::endl << std::endl;
                // Each cascade uses a different set of pixels for its features.  We compute
                // their representations relative to the initial shape first.
                std::vector<unsigned long> anchor_idx;
                std::vector<Point2f > deltas;
                create_shape_relative_encoding(initial_shape, pixel_coordinates[cascade], anchor_idx, deltas);

                // First compute the feature_pixel_values for each training sample at this
                // level of the cascade.
                for (unsigned long i = 0; i < samples.size(); ++i)
                {
                    extract_feature_pixel_values(*images[samples[i].image_idx], samples[i].rect,
                        samples[i].current_shape, initial_shape, anchor_idx,
                        deltas, samples[i].feature_pixel_values);
                }

                // Now start building the trees at this cascade level.
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

    private:

        static Mat object_to_shape (
            const FullObjectDetection& obj
        )
        {
			// create an matrix of 2xN, where N is the amount of parts
            Mat shape(2, obj.num_parts(), CV_64F);
std::cout << obj.get_rect() << std::endl;
            const point_transform_affine tform_from_img = impl::normalizing_tform(obj.get_rect());

            for (unsigned long i = 0; i < obj.num_parts(); ++i)
            {
				Point2f w = obj.part(i);
				w.x = cvRound(w.x);
				w.y = cvRound(w.y);

                Point2f p = tform_from_img(/*obj.part(i)*/w);
                shape.at<double>(0, i) = p.x;
                shape.at<double>(1, i) = p.y;
            }

            return shape;
        }

struct TrainingSample
{
	/*!

	CONVENTION
		- feature_pixel_values.size() == get_feature_pool_size()
		- feature_pixel_values[j] == the value of the j-th feature pool
		  pixel when you look it up relative to the shape in current_shape.

		- target_shape == The truth shape.  Stays constant during the whole
		  training process.
		- rect == the position of the object in the image_idx-th image.  All shape
		  coordinates are coded relative to this rectangle.
	!*/

	unsigned long image_idx;
	Rect rect;
	Mat target_shape;

	Mat current_shape;
	std::vector<float> feature_pixel_values;

	void swap(TrainingSample& item)
	{
		std::swap(image_idx, item.image_idx);
		std::swap(rect, item.rect);
		cv::swap(target_shape, item.target_shape);
		cv::swap(current_shape, item.current_shape);
		feature_pixel_values.swap(item.feature_pixel_values);
	}
};

        impl::RegressionTree make_regression_tree (
            std::vector<TrainingSample>& samples,
            const std::vector<Point2f >& pixel_coordinates
        ) const
        {
            using namespace impl;
            std::deque<std::pair<unsigned long, unsigned long> > parts;
            parts.push_back(std::make_pair(0, (unsigned long)samples.size()));

            impl::RegressionTree tree;

            // walk the tree in breadth first order
            const unsigned long num_split_nodes = static_cast<unsigned long>(std::pow(2.0, (double)get_tree_depth())-1);

            std::vector<Mat > sums(num_split_nodes*2+1);
            for (unsigned long i = 0; i < sums.size(); ++i)
				sums[i] = cv::Mat::zeros(samples[0].current_shape.rows, samples[0].current_shape.cols, samples[0].current_shape.type());

            for (unsigned long i = 0; i < samples.size(); ++i)
            {
				sums[0] += samples[i].target_shape - samples[i].current_shape;
			}

            for (unsigned long i = 0; i < num_split_nodes; ++i)
            {
                std::pair<unsigned long,unsigned long> range = parts.front();
                parts.pop_front();

                const impl::SplitFeature split = generate_split(samples, range.first,
                    range.second, pixel_coordinates, sums[i], sums[left_child(i)],
                    sums[right_child(i)]);
                tree.splits.push_back(split);
//std::cout << "Split #" << i << " = " << split.thresh << std::endl;
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
                    tree.leaf_values[i] = sums[num_split_nodes+i]*get_nu()/(parts[i].second - parts[i].first);
                else
                    tree.leaf_values[i] = Mat::zeros(samples[0].target_shape.rows, samples[0].target_shape.cols, CV_64F);//zeros_matrix(samples[0].target_shape);

                // now adjust the current shape based on these predictions
                for (unsigned long j = parts[i].first; j < parts[i].second; ++j)
                    samples[j].current_shape += tree.leaf_values[i];
            }

            return tree;
        }

		/**
		 * Create an split feature with randomly generated threshold.
		 */
        impl::SplitFeature randomly_generate_split_feature (
            const std::vector<Point2f >& pixel_coordinates
        ) const
        {
            const double lambda = get_lambda();
            impl::SplitFeature feat;
            double accept_prob;
            do
            {
                feat.idx1   = getRnd().uniform(0, 0x0FFFFFFF)%get_feature_pool_size();
                feat.idx2   = getRnd().uniform(0, 0x0FFFFFFF)%get_feature_pool_size();
                const double dist = dlib::impl::length(pixel_coordinates[feat.idx1]-pixel_coordinates[feat.idx2]);
                accept_prob = std::exp(-dist/lambda);
            }
            while(feat.idx1 == feat.idx2 || !(accept_prob > getRnd().uniform(0.0, 1.0)));

            feat.thresh = (getRnd().uniform(0.0, 1.0)*256 /*- 128*/)/2.0;

            return feat;
        }


		/**
		 * Generate a bunch of random splits, test them and return the best one.
		 */
        impl::SplitFeature generate_split (
            const std::vector<TrainingSample>& samples,
            unsigned long begin,
            unsigned long end,
            const std::vector<Point2f >& pixel_coordinates,
            const Mat& sum,
            Mat& left_sum,
            Mat& right_sum
        ) const
        {
            const unsigned long num_test_splits = get_num_test_splits();

            // sample the random features we test in this function
            std::vector<impl::SplitFeature> feats;
            feats.reserve(num_test_splits);
            for (unsigned long i = 0; i < num_test_splits; ++i)
                feats.push_back(randomly_generate_split_feature(pixel_coordinates));

            std::vector<Mat > left_sums(num_test_splits);
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
							left_sums[i] = temp;
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
                right_sum = sum - left_sum;
            }
            else
            {
                right_sum = sum;
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
        unsigned long partition_samples (
            const impl::SplitFeature& split,
            std::vector<TrainingSample>& samples,
            unsigned long begin,
            unsigned long end
        ) const
        {
            //

            unsigned long i = begin;
            for (unsigned long j = begin; j < end; ++j)
            {
                if (samples[j].feature_pixel_values[split.idx1] - samples[j].feature_pixel_values[split.idx2] > split.thresh)
                {
                    samples[i].swap(samples[j]);
                    ++i;
                }
            }
            return i;
        }



        Mat populate_training_sample_shapes(
            const std::vector<std::vector<FullObjectDetection*> >& objects,
            std::vector<TrainingSample>& samples
        ) const
        {
            samples.clear();
            Mat mean_shape;
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
std::cout << sample.target_shape << std::endl;
                    for (unsigned long itr = 0; itr < get_oversampling_amount(); ++itr)
                        samples.push_back(sample);

                    // sum the current shape to the mean shape
                    if (mean_shape.rows == 0)
						mean_shape = sample.target_shape;
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
                if ((i%get_oversampling_amount()) == 0)
                {
                    // The mean shape is what we really use as an initial shape so always
                    // include it in the training set as an example starting shape.
                    samples[i].current_shape = mean_shape;
                }
                else
                {
                    // Pick a random convex combination of two of the target shapes and use
                    // that as the initial shape for this sample.
                    const unsigned long rand_idx = getRnd().uniform(0, 0x0FFFFFFF) % samples.size();
                    const unsigned long rand_idx2 = getRnd().uniform(0, 0x0FFFFFFF) % samples.size();
                    const double alpha = getRnd().uniform(0.0, 1.0);
                    samples[i].current_shape = alpha*samples[rand_idx].target_shape + (1-alpha)*samples[rand_idx2].target_shape;
                }
            }

            return mean_shape;
        }


        void randomly_sample_pixel_coordinates (
            std::vector<Point2f>& pixel_coordinates,
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
                pixel_coordinates[i].x = (max_x-min_x) + min_x;//getRnd().uniform(0.0, 1.0)*(max_x-min_x) + min_x;
                pixel_coordinates[i].y = (max_y-min_y) + min_y;//getRnd().uniform(0.0, 1.0)*(max_y-min_y) + min_y;
            }
        }

        std::vector<std::vector<Point2f > > randomly_sample_pixel_coordinates (
            const Mat& initial_shape
        ) const
        {
            const double padding = get_feature_pool_region_padding();
            // Figure figure out the bounds on the object shapes.  We will sample uniformly
            // from this box.

			double min_x, max_x;
			cv::minMaxLoc( initial_shape( Range(0, 1), Range::all() ), &min_x, &max_x );
			min_x -= padding;
			max_x -= padding;

			double min_y, max_y;
			cv::minMaxLoc( initial_shape( Range(1, 2), Range::all() ), &min_y, &max_y );
			min_y += padding;
			max_y += padding;

            std::vector<std::vector<Point2f > > pixel_coordinates;
            pixel_coordinates.resize(get_cascade_depth());
            for (unsigned long i = 0; i < get_cascade_depth(); ++i)
                randomly_sample_pixel_coordinates(pixel_coordinates[i], min_x, min_y, max_x, max_y);
            return pixel_coordinates;
        }





        unsigned long _cascade_depth;
        unsigned long _tree_depth;
        unsigned long _num_trees_per_cascade_level;
        double _nu;
        unsigned long _oversampling_amount;
        unsigned long _feature_pool_size;
        double _lambda;
        unsigned long _num_test_splits;
        double _feature_pool_region_padding;
        bool _verbose;
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------


    double test_shape_predictor (
        const ShapePredictor& sp,
        const std::vector<Mat*>& images,
        const std::vector<std::vector<FullObjectDetection*> >& objects,
        const std::vector<std::vector<double> >& scales
    )
    {
        // make sure requires clause is not broken
#ifdef ENABLE_ASSERTS
        /*DLIB_CASSERT( images.size() == objects.size() ,
            "\t double test_shape_predictor()"
            << "\n\t Invalid inputs were given to this function. "
            << "\n\t images.size():  " << images.size()
            << "\n\t objects.size(): " << objects.size()
        );*/
        for (unsigned long i = 0; i < objects.size(); ++i)
        {
            for (unsigned long j = 0; j < objects[i].size(); ++j)
            {
                /*DLIB_CASSERT(objects[i][j].num_parts() == sp.num_parts(),
                    "\t double test_shape_predictor()"
                    << "\n\t Invalid inputs were given to this function. "
                    << "\n\t objects["<<i<<"]["<<j<<"].num_parts(): " << objects[i][j].num_parts()
                    << "\n\t sp.num_parts(): " << sp.num_parts()
                );*/
            }
            if (scales.size() != 0)
            {
                /*DLIB_CASSERT(objects[i].size() == scales[i].size(),
                    "\t double test_shape_predictor()"
                    << "\n\t Invalid inputs were given to this function. "
                    << "\n\t objects["<<i<<"].size(): " << objects[i].size()
                    << "\n\t scales["<<i<<"].size(): " << scales[i].size()
                );*/

            }
        }
#endif

        //running_stats<double> rs;
        double rs = 0;
        int count = 0;
        for (unsigned long i = 0; i < objects.size(); ++i)
        {
            for (unsigned long j = 0; j < objects[i].size(); ++j)
            {
                // Just use a scale of 1 (i.e. no scale at all) if the caller didn't supply
                // any scales.
                //const double scale = scales.size()==0 ? 1 : scales[i][j];
                const double scale = 1;

                FullObjectDetection det = sp.detect(*images[i], objects[i][j]->get_rect());
std::cout << *objects[i][j] << std::endl;
std::cout << det << std::endl;


                for (unsigned long k = 0; k < det.num_parts(); ++k)
                {
					Point2f gold, fit;
					fit.x = round( det.part(k).x );
					fit.y = round( det.part(k).y );
					gold.x = round( objects[i][j]->part(k).x );
					gold.y = round( objects[i][j]->part(k).y );
                    double score = dlib::impl::length(fit - gold)/scale;
                    rs += score;
                    ++count;
                    //rs.add(score);
                }
            }
        }
        //return rs.mean();
        return rs / count;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_array
        >
    double test_shape_predictor (
        const ShapePredictor& sp,
        const std::vector<Mat*>& images,
        const std::vector<std::vector<FullObjectDetection*> >& objects
    )
    {
        std::vector<std::vector<double> > no_scales;
        return test_shape_predictor(sp, images, objects, no_scales);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SHAPE_PREDICToR_H_

