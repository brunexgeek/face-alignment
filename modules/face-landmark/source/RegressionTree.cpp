#include <ert/RegressionTree.hh>


namespace ert {


// a tree is just a std::vector<impl::SplitFeature>.  We use this function to navigate the
// tree nodes
unsigned long left_child (unsigned long idx) { return 2*idx + 1; }
/*!
	ensures
	- returns the index of the left child of the binary tree node idx
!*/
unsigned long right_child (unsigned long idx) { return 2*idx + 2; }


void SplitFeature::serialize( std::ostream &out ) const
{
	Serializable::serialize(out, idx1);
	Serializable::serialize(out, idx2);
	Serializable::serialize(out, thresh);
}


void SplitFeature::deserialize( std::istream &in )
{
	Serializable::deserialize(in, idx1);
	Serializable::deserialize(in, idx2);
	Serializable::deserialize(in, thresh);
}


const cv::Mat& RegressionTree::operator()(
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


void RegressionTree::serialize( std::ostream &out ) const
{
	// serialize the splits
	Serializable::serialize(out, splits.size());
	for (size_t i = 0; i < splits.size(); ++i)
	{
		splits[i].serialize(out);
	}

	// serialize the leaf node matrices
	Serializable::serialize(out, leaf_values.size());
	for (size_t i = 0; i < leaf_values.size(); ++i)
	{
		Serializable::serialize(out, leaf_values[i]);
	}
}


void RegressionTree::deserialize( std::istream &in )
{
	size_t entries;
	// deserialize the splits
	Serializable::deserialize(in, entries);
	splits.resize(entries);
	for (size_t i = 0; i < splits.size(); ++i)
	{
		splits[i].deserialize(in);
	}

	// deserialize the leaf node matrices
	Serializable::deserialize(in, entries);
	leaf_values.resize(entries);
	for (size_t i = 0; i < leaf_values.size(); ++i)
	{
		Serializable::deserialize(in, leaf_values[i]);
	}
}


};
