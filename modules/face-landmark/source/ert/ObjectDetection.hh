// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FULL_OBJECT_DeTECTION_Hh_
#define DLIB_FULL_OBJECT_DeTECTION_Hh_


#include <vector>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <fstream>


namespace ert {


using namespace cv;



class ObjectDetection
{
	public:
		const static Point2f OBJECT_PART_NOT_PRESENT;

		const static uint16_t OPEN = 0xFFF0;

		const static uint16_t CLOSE = 0xFFFC;

		const static uint16_t END = 0xFFFE;

		ObjectDetection( const Rect& rect_, const std::vector<Point2f>& parts_ );

		ObjectDetection( const std::string &fileName );

		ObjectDetection();

		explicit ObjectDetection(
			const Rect& rect );

		const Rect& get_rect() const
		{
			return rect;
		}

		Rect& get_rect()
		{
			return rect;
		}

		void set_rect(
			Rect &rect )
		{
			this->rect = rect;
		}

		unsigned long num_parts() const
		{
			return parts.size();
		}

		const Point2f& part( unsigned long idx ) const
		{
			// make sure requires clause is not broken
			/*DLIB_ASSERT(idx < num_parts(),
				"\t point full_object_detection::part()"
				<< "\n\t Invalid inputs were given to this function "
				<< "\n\t idx:         " << idx
				<< "\n\t num_parts(): " << num_parts()
				<< "\n\t this:        " << this
				);*/
			return parts[idx];
		}


		void computeBoundingBox(
			float border );

		void save(
			const std::string& fileName ) const;

		void load(
			const std::string &fileName );

		bool isAllPartsInRect (
			const Rect& area ) const;

		void plot(
			cv::Mat &output,
			const uint16_t *layout,
			const Scalar &color ) const;

		ObjectDetection &operator/=( float factor );

		ObjectDetection &operator-=( const Point2f &factor );

		void remove( size_t from, size_t to );

	private:
		Rect rect;
		std::vector<Point2f> parts;

		void loadPoints(
			const std::string &fileName );
};

}

#endif // DLIB_FULL_OBJECT_DeTECTION_H_

