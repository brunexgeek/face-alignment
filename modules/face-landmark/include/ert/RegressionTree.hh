#ifndef REGRESSION_TREE
#define REGRESSION_TREE


#include <opencv2/opencv.hpp>
#include <ert/Serializable.hh>


namespace ert {


	using namespace cv;


	// a tree is just a std::vector<impl::SplitFeature>.  We use this function to navigate the
	// tree nodes
	unsigned long left_child (unsigned long idx);
	/*!
		ensures
			- returns the index of the left child of the binary tree node idx
	!*/
	unsigned long right_child (unsigned long idx);



	struct SplitFeature : public Serializable
	{
		uint16_t idx1;
		uint16_t idx2;
		float thresh;

		void serialize( std::ostream &out ) const;

		void deserialize( std::istream &in );
	};


	/*!
		ensures
			- returns the index of the left child of the binary tree node idx
	!*/

	class RegressionTree : public Serializable
	{
		public:
			std::vector<SplitFeature> splits;
			std::vector<cv::Mat> leaf_values;

			const cv::Mat& operator()(
				const std::vector<double>& feature_pixel_values
			) const;

			void serialize( std::ostream &out ) const;

			void deserialize( std::istream &in );

	};

};

#endif //  REGRESSION_TREE
