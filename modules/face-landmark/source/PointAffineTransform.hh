// Copyright (C) 2003  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_POINT_TrANSFORMS_H_
#define DLIB_POINT_TrANSFORMS_H_

#include "point_transforms_abstract.h"
#include <vector>
#include <ert/Serializable.hh>
#include <opencv2/opencv.hpp>
#include <ert/opencv.hh>


namespace ert
{


	using namespace cv;

	double length_squared( const Point2f &p );


	// CHECKED!!!
    class PointTransformAffine
    {
    public:

        PointTransformAffine (
        )
        {
            //m = identity_matrix<double>(2);
            b.x = 0;
            b.y = 0;
        }

        PointTransformAffine (
            const Mat& m_,
            const Point2f& b_
        ) :m(m_), b(b_)
        {
        }


        PointTransformAffine (
            const Mat& m_,
            const Mat& b_
        ) :m(m_)
        {
			b = Point2f( b_.at<double>(0,0), b_.at<double>(1,0) );
			/*std::cout << "m_ = " << m_ << std::endl;
			std::cout << "b_ = " << b_ << std::endl;
			std::cout << "b = " << b << std::endl;*/
        }

        const Point2f operator() (
            const Point2f& p
        ) const;

        const Mat& get_m(
        ) const { return m; }

        const Point2f& get_b(
        ) const { return b; }

		static PointTransformAffine findAffineTransform (
			const std::vector<Point2f>& from_points,
			const std::vector<Point2f>& to_points
		);

		static PointTransformAffine findSimilarityTransform (
			const std::vector<Point2f>& from_points,
			const std::vector<Point2f>& to_points
		);

    private:
        Mat m;
        Point2f b;
    };

// ----------------------------------------------------------------------------------------


	PointTransformAffine operator* (
		const PointTransformAffine& lhs,
		const PointTransformAffine& rhs
	);


}

#endif // DLIB_POINT_TrANSFORMS_H_

