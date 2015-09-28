#ifndef REGRESSION_TREE
#define REGRESSION_TREE


#include <opencv2/opencv.hpp>
#include "Serializable.hh"


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
		unsigned long idx1;
		unsigned long idx2;
		float thresh;

		private:
			void serialize( std::ostream &out )
			{
				}
			 void deserialize( std::istream &in )
			 {
			 }
	};



	/*!
		ensures
			- returns the index of the left child of the binary tree node idx
	!*/

	struct RegressionTree
	{
		std::vector<SplitFeature> splits;
		std::vector<cv::Mat> leaf_values;

		inline const cv::Mat& operator()(
			const std::vector<double>& feature_pixel_values
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


	};

};

#endif //  REGRESSION_TREE
