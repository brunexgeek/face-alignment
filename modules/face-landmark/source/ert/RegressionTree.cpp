#include "RegressionTree.hh"


namespace ert {



// a tree is just a std::vector<impl::SplitFeature>.  We use this function to navigate the
// tree nodes
unsigned long left_child (unsigned long idx) { return 2*idx + 1; }
/*!
	ensures
	- returns the index of the left child of the binary tree node idx
!*/
unsigned long right_child (unsigned long idx) { return 2*idx + 2; }



};
