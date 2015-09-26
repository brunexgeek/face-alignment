// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FULL_OBJECT_DeTECTION_Hh_
#define DLIB_FULL_OBJECT_DeTECTION_Hh_

#include "full_object_detection_abstract.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <cstdio>

namespace dlib
{


using namespace cv;

// ----------------------------------------------------------------------------------------

    const static Point2f OBJECT_PART_NOT_PRESENT(0x7FFFFFFF,
                                                0x7FFFFFFF);

// ----------------------------------------------------------------------------------------

    class FullObjectDetection
    {
		public:

			FullObjectDetection( const Rect& rect_, const std::vector<Point2f>& parts_ )
			{
				rect = rect_;
				parts = parts_;
			}

			FullObjectDetection( const char *fileName )
			{
				loadPointsFile(fileName);
				computeBoundingBox(0.1);
			}

			FullObjectDetection()
			{
			}

			explicit FullObjectDetection(
				const Rect& rect_
			) : rect(rect_)
			{
			}

			const Rect& get_rect() const { return rect; }
			Rect& get_rect() { return rect; }
			unsigned long num_parts() const { return parts.size(); }


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

			friend void serialize (
				const FullObjectDetection& item,
				std::ostream& out
			)
			{
				//int version = 1;
				/*serialize(version, out);
				serialize(item.rect, out);
				serialize(item.parts, out);*/
			}

			friend void deserialize (
				FullObjectDetection& item,
				std::istream& in
			)
			{
				//int version = 0;
				/*deserialize(version, in);
				if (version != 1)
					throw serialization_error("Unexpected version encountered while deserializing dlib::full_object_detection.");

				deserialize(item.rect, in);
				deserialize(item.parts, in);*/
			}


			void computeBoundingBox(
				float border )
			{
				float x, y;
				float minX = 100000, minY = 100000, maxX = -100, maxY = -100;
				Rect bbox = Rect(0, 0, 0, 0);

				for (size_t i = 0; i < parts.size(); ++i)
				{
					x = parts[i].x;
					y = parts[i].y;
					if (minX > x) minX = x;
					if (minY > y) minY = y;
					if (maxX < x) maxX = x;
					if (maxY < y) maxY = y;
				}

				bbox.x = minX;
				bbox.y = minY;
				bbox.width = maxX - minX;
				bbox.height = maxY - minY;

				bbox.x -= (float(bbox.width) * border);
				bbox.y -= (float(bbox.height) * border);
				bbox.width += (float(bbox.width) * (border * 2));
				bbox.height += (float(bbox.height) * (border * 2));

				printf("{ Pos: %d x %d    Size : %d x %d }\n", bbox.x, bbox.y, bbox.width, bbox.height);

				this->rect = bbox;
			}


			void loadPointsFile( const char *fileName )
			{
				char *line;
				size_t len = 0;
				float x, y;
				int lines = 0;
				FILE *fp;
				int p;
				char s[32];

				fp = fopen(fileName, "rt");
				if (fp == NULL) return;

				while(!feof(fp))
				{
					if ( getline(&line, &len, fp) < 0) continue;

					/*if ( points == NULL && strstr(line, "n_points:") != NULL )
					{
						if (sscanf(line, "%s %i", s, &p) == 2)
							points = new std::vector<Point2f>(p);
					}*/

					if ( sscanf(line, "%f %f", &x, &y) == 2)
					{
						parts.push_back( Point2f( std::ceil(x), std::ceil(y) ) );
						lines++;
					}
				}
				fclose(fp);
			}

		private:
			Rect rect;
			std::vector<Point2f> parts;
    };

// ----------------------------------------------------------------------------------------

    inline bool all_parts_in_rect (
        const FullObjectDetection& obj
    )
    {
        for (unsigned long i = 0; i < obj.num_parts(); ++i)
        {
            if (obj.get_rect().contains(obj.part(i)) == false &&
                obj.part(i) != OBJECT_PART_NOT_PRESENT)
                return false;
        }
        return true;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FULL_OBJECT_DeTECTION_H_

