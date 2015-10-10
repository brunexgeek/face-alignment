#ifndef ShAPE_PREDICTOR_TRAINER
#define ShAPE_PREDICTOR_TRAINER


#include <opencv2/opencv.hpp>
#include "ShapePredictor.hh"
#include "PointAffineTransform.hh"
#include "RegressionTree.hh"


namespace ert {


	using namespace cv;


	class ShapePredictor;


	double length( const Point2f &p );

	Point2f location (
		const Mat& shape,
		unsigned long idx
	);


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
		cv::Rect rect;
		cv::Mat target_shape;

		cv::Mat current_shape;
		std::vector<double> feature_pixel_values;

		void swap(TrainingSample& item)
		{
			std::swap(image_idx, item.image_idx);
			std::swap(rect, item.rect);
			cv::swap(target_shape, item.target_shape);
			cv::swap(current_shape, item.current_shape);
			feature_pixel_values.swap(item.feature_pixel_values);
		}
	};

		PointTransformAffine normalizing_tform (
			const Rect& rect
		);


	class ShapePredictorTrainer
	{
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


			std::string get_random_seed (
			) const { return rnd.get_seed(); }

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
				const std::vector<cv::Mat*>& images,
				const std::vector<std::vector<ObjectDetection*> >& objects ) const;

		private:


			// CHECKED!!!
			static cv::Mat object_to_shape (
				const ObjectDetection& obj
			)
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


			void printShape( const std::string& prefix, const cv::Mat& mat ) const
			{
				std::cout << prefix;
				for (int i = 0; i < mat.cols; ++i)
					for (int j = 0; j < mat.rows; ++j)
						std::cout << mat.at<double>(j,i) << std::endl;
				std::cout << std::endl;
			}


			RegressionTree make_regression_tree (
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
			SplitFeature randomly_generate_split_feature (
				const std::vector<cv::Point2f >& pixel_coordinates
			) const
			{
				const double lambda = get_lambda();
				SplitFeature feat;
				double accept_prob;
				do
				{
					feat.idx1   = rnd.get_random_32bit_number() % get_feature_pool_size();
					feat.idx2   = rnd.get_random_32bit_number() % get_feature_pool_size();
					const double dist = length(pixel_coordinates[feat.idx1]-pixel_coordinates[feat.idx2]);
					accept_prob = std::exp(-dist/lambda);
				}
				while(feat.idx1 == feat.idx2 || !(accept_prob > rnd.get_random_double()));

				feat.thresh = (rnd.get_random_double()*256 - 128)/2.0;
				//feat.thresh = (rnd.get_random_double()*256 - 128)/2.0 + 128;

				return feat;
			}


			/**
			 * Generate a bunch of random splits, test them and return the best one.
			 */
			SplitFeature generate_split (
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
			unsigned long partition_samples (
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



			cv::Mat populate_training_sample_shapes(
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
					if ((i%get_oversampling_amount()) == 0)
					{
						// The mean shape is what we really use as an initial shape so always
						// include it in the training set as an example starting shape.
						mean_shape.copyTo(samples[i].current_shape);
					}
					else
					{
						// Pick a random convex combination of two of the target shapes and use
						// that as the initial shape for this sample.
						const unsigned long rand_idx = rnd.get_random_32bit_number() % samples.size();
						const unsigned long rand_idx2 = rnd.get_random_32bit_number() % samples.size();
						const double alpha = rnd.get_random_double();
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


			void randomly_sample_pixel_coordinates (
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
					pixel_coordinates[i].x = rnd.get_random_double()*(max_x-min_x) + min_x;
					pixel_coordinates[i].y = rnd.get_random_double()*(max_y-min_y) + min_y;
	//std::cout << "pixel_coordinates[" << i << "] = " << pixel_coordinates[i] << std::endl;
	//std::getchar();
				}
			}

			std::vector<std::vector<cv::Point2f > > randomly_sample_pixel_coordinates (
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



			mutable dlib::rand rnd;

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


};


#endif  // ShAPE_PREDICTOR_TRAINER
