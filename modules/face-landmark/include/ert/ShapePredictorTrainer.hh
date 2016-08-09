#ifndef ShAPE_PREDICTOR_TRAINER
#define ShAPE_PREDICTOR_TRAINER


#include <opencv2/opencv.hpp>
#include <ert/ShapePredictor.hh>
#include <ert/RegressionTree.hh>


namespace ert {


	using namespace cv;


	class Random;

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

		/*PointTransformAffine normalizing_tform (
			const Rect& rect
		);*/


	class ShapePredictorTrainer
	{
		public:

			ShapePredictorTrainer ( );

			unsigned long get_cascade_depth () const;

			void set_cascade_depth (
				unsigned long depth);

			unsigned long get_tree_depth () const;

			void set_tree_depth (
				unsigned long depth);

			unsigned long get_num_trees_per_cascade_level () const;

			void set_num_trees_per_cascade_level (
				unsigned long num);

			double get_nu () const;
			void set_nu (
				double nu);


			std::string get_random_seed () const;

			unsigned long get_oversampling_amount () const;

			void set_oversampling_amount (
				unsigned long amount);

			unsigned long get_feature_pool_size () const;

			void set_feature_pool_size (
				unsigned long size);

			double get_lambda () const;

			void set_lambda (
				double lambda);

			unsigned long get_num_test_splits (
			) const;
			void set_num_test_splits (
				unsigned long num);


			double get_feature_pool_region_padding (
			) const;
			void set_feature_pool_region_padding (
				double padding);

			void be_verbose ();

			void be_quiet ();

			ShapePredictor train (
				const std::vector<cv::Mat*>& images,
				const std::vector<std::vector<ObjectDetection*> >& objects ) const;

		private:


			// CHECKED!!!
			static cv::Mat object_to_shape (
				const ObjectDetection& obj
			);


			void printShape( const std::string& prefix, const cv::Mat& mat ) const;


			RegressionTree make_regression_tree (
				std::vector<TrainingSample>& samples,
				const std::vector<cv::Point2f >& pixel_coordinates
			) const;

			/**
			 * Create an split feature with randomly generated threshold.
			 */
			SplitFeature randomly_generate_split_feature (
				const std::vector<cv::Point2f >& pixel_coordinates
			) const;


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
			) const;

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
			) const;



			cv::Mat populate_training_sample_shapes(
				const std::vector<std::vector<ObjectDetection*> >& objects,
				std::vector<TrainingSample>& samples
			) const;


			void randomly_sample_pixel_coordinates (
				std::vector<cv::Point2f>& pixel_coordinates,
				const double min_x,
				const double min_y,
				const double max_x,
				const double max_y
			) const;

			std::vector<std::vector<cv::Point2f > > randomly_sample_pixel_coordinates (
				const cv::Mat& initial_shape ) const;

			Random *rnd;

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
