#include "ObjectDetection.hh"


namespace ert {


using namespace cv;


const Point2f ObjectDetection::OBJECT_PART_NOT_PRESENT(0x7FFFFFFF, 0x7FFFFFFF);


ObjectDetection::ObjectDetection( const Rect& rect_, const std::vector<Point2f>& parts_ )
{
	rect = rect_;
	parts = parts_;
}

ObjectDetection::ObjectDetection( const std::string &fileName )
{
	loadPoints(fileName);
	computeBoundingBox(0.1);
}

ObjectDetection::ObjectDetection()
{
}

ObjectDetection::ObjectDetection(
	const Rect& rect
) : rect(rect)
{
}

void ObjectDetection::computeBoundingBox(
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

	/*printf("{ Pos: %d x %d    Size : %d x %d    End: %d x %d }\n",
		bbox.x, bbox.y, bbox.width, bbox.height, bbox.x + bbox.width,
		bbox.y + bbox.height);*/

	this->rect = bbox;
}


void ObjectDetection::save( const std::string& fileName ) const
{
	std::ofstream out(fileName.c_str());

	out << "version: 1" << std::endl;
	out << "n_points: " << num_parts() << std::endl;
	out << "{" << std::endl;

	for (int i = 0; i < (int)num_parts(); ++i)
	{
		const Point2f &point = part(i);
		out << floor(point.x) << " " << floor(point.y) << std::endl;
	}

	out << "}";
	out.close();
}

void ObjectDetection::loadPoints( const std::string &fileName )
{
	char *line;
	size_t len = 0;
	float x, y;
	int lines = 0;
	FILE *fp;
	//int p;
	//char s[32];

	fp = fopen(fileName.c_str(), "rt");
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
			parts.push_back( Point2f( std::floor(x), std::floor(y) ) );
			lines++;
		}
	}
	fclose(fp);
}

void ObjectDetection::load(
	const std::string &fileName )
{
	loadPoints(fileName.c_str());
	computeBoundingBox(0.1);
}


bool ObjectDetection::isAllPartsInRect (
	const Rect& area ) const
    {
        for (unsigned long i = 0; i < num_parts(); ++i)
        {
			const Point2f &point = parts[i];
            if (point != ObjectDetection::OBJECT_PART_NOT_PRESENT &&
                area.contains(point) == false)
                return false;
        }
        return true;
    }


void ObjectDetection::plot(
	cv::Mat &output,
	const uint16_t *layout,
	const Scalar &color ) const
{
	uint16_t first, toindex;

	if (layout == NULL || output.rows == 0) return;
	first = layout[0];

	for (int i = 0; layout[i] != ObjectDetection::END; ++i)
	{
		if (layout[i] == ObjectDetection::CLOSE || layout[i] == ObjectDetection::OPEN)
			continue;

		toindex = layout[i+1];

		// check if we need to plot a closed polygon
		if (toindex == ObjectDetection::CLOSE)
		{
			toindex = first;
			first = layout[i+2];
		}
		else
		if (toindex == ObjectDetection::OPEN)
		{
			first = layout[i+2];
			continue;
		}

		// round the points
		Point2f from = parts[ layout[i] ];
		from.x = round(from.x);
		from.y = round(from.y);
		Point2f to = parts[toindex];
		to.x = round(to.x);
		to.y = round(to.y);
		// plot the line between the previous part and the current one
		cv::line(output, from, to, color, 1);
	}
}


ObjectDetection &ObjectDetection::operator/=( float factor )
{
	// rescale the points
	for (unsigned long i = 0; i < num_parts(); ++i)
	{
		parts[i].x /= factor;
		parts[i].y /= factor;
	}
	// rescale the bounding box
	rect.x /= factor;
	rect.y /= factor;
	rect.width /= factor;
	rect.height /= factor;

	return *this;
}


ObjectDetection &ObjectDetection::operator-=( const Point2f &factor )
{
	// rescale the points
	for (unsigned long i = 0; i < num_parts(); ++i)
	{
		parts[i].x -= factor.x;
		parts[i].y -= factor.y;
	}
	// rescale the bounding box
	rect.x -= factor.x;
	rect.y -= factor.y;

	return *this;
}


void ObjectDetection::remove( size_t from, size_t to )
{
	if (from < 0 || from > num_parts() || to < 0 || to > num_parts() || from <= to)
		return;

	parts.erase(parts.begin() + from, parts.begin() + to);
}


} // namespace ert
