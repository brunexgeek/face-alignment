#include "ShapePredictor.hh"


namespace ert {


void create_shape_relative_encoding (
const Mat& shape,
const std::vector<Point2f> &pixel_coordinates,
std::vector<unsigned long>& anchor_idx,
std::vector<Point2f>& deltas
);


void extract_feature_pixel_values (
	const Mat& img,
	const Rect& rect,
	const Mat& current_shape,
	const Mat& reference_shape,
	const std::vector<unsigned long>& reference_pixel_anchor_idx,
	const std::vector<Point2f>& reference_pixel_deltas,
	std::vector<double>& feature_pixel_values
);


PointTransformAffine unnormalizing_tform (
	const Rect& rect
);




void plotFace(
	Mat &image,
	ObjectDetection &det,
	const Scalar &color,
	int thickness = 1 )
{
	{
		// contorno da face
		for (size_t i = 0; i < 16; ++i)
			cv::line(image, det.part(i), det.part(i+1), color, thickness);
		// sobrancelha esquerda
		for (size_t i = 17; i < 21; ++i)
			cv::line(image, det.part(i), det.part(i+1), color, thickness);
		// sonbrancelha direita
		for (size_t i = 22; i < 26; ++i)
			cv::line(image, det.part(i), det.part(i+1), color, thickness);
		// linha vertical do nariz
		for (size_t i = 27; i < 30; ++i)
			cv::line(image, det.part(i), det.part(i+1), color, thickness);
		// linha horizontal do nariz
		for (size_t i = 31; i < 35; ++i)
			cv::line(image, det.part(i), det.part(i+1), color, thickness);
		// olho esquerdo
		for (size_t i = 36; i < 41; ++i)
			cv::line(image, det.part(i), det.part(i+1), color, thickness);
		cv::line(image, det.part(41), det.part(36), color, thickness);
		// olho direito
		for (size_t i = 42; i < 47; ++i)
			cv::line(image, det.part(i), det.part(i+1), color, thickness);
		cv::line(image, det.part(47), det.part(42), color, thickness);
		// parte externa da boca
		for (size_t i = 48; i < 59; ++i)
			cv::line(image, det.part(i), det.part(i+1), color, thickness);
		cv::line(image, det.part(59), det.part(48), color, thickness);
		// parte externa da boca
		for (size_t i = 60; i < 67; ++i)
			cv::line(image, det.part(i), det.part(i+1), color, thickness);
		cv::line(image, det.part(67), det.part(60), color, thickness);
	}
}


void printRow( std::ostream& os, bool isX, const ObjectDetection& obj )
{
	int c;
	bool f;

	if (isX)
		os << "X = [ ";
	else
		os << "Y = [ ";

	for (c = 0, f = false; c < (int)obj.num_parts(); ++c)
	{
		//if (obj.num_parts() > 20)
		{
			//if ((c < 10) || (c > (obj.num_parts() - 10)))
				if (isX)
					os << round(obj.part(c).x) << " ";
				else
					os << round(obj.part(c).y) << " ";
			/*else
			{
				if (!f)
				{
					std::cout << " ... ";
					f = true;
				}
			}*/
		}
	}
	os << "]" << std::endl;
}


std::ostream& operator<<(std::ostream& os, const ObjectDetection& obj)
{
	printRow(os, true, obj);
	printRow(os, false, obj);
	return os;
}



	double mylength( const Point2f &p )
	{
		return std::sqrt( (p.x * p.x) + (p.y * p.y) );
		//return (double)p.x + (double)p.y;
	}


    double test_shape_predictor (
        const ShapePredictor& sp,
        const std::vector<Mat*>& images,
        const std::vector<std::vector<ObjectDetection*> >& objects,
        const std::vector<std::vector<double> >& scales
    )
    {
        // make sure requires clause is not broken
#ifdef ENABLE_ASSERTS
        /*DLIB_CASSERT( images.size() == objects.size() ,
            "\t double test_shape_predictor()"
            << "\n\t Invalid inputs were given to this function. "
            << "\n\t images.size():  " << images.size()
            << "\n\t objects.size(): " << objects.size()
        );*/
        for (unsigned long i = 0; i < objects.size(); ++i)
        {
            for (unsigned long j = 0; j < objects[i].size(); ++j)
            {
                /*DLIB_CASSERT(objects[i][j].num_parts() == sp.num_parts(),
                    "\t double test_shape_predictor()"
                    << "\n\t Invalid inputs were given to this function. "
                    << "\n\t objects["<<i<<"]["<<j<<"].num_parts(): " << objects[i][j].num_parts()
                    << "\n\t sp.num_parts(): " << sp.num_parts()
                );*/
            }
            if (scales.size() != 0)
            {
                /*DLIB_CASSERT(objects[i].size() == scales[i].size(),
                    "\t double test_shape_predictor()"
                    << "\n\t Invalid inputs were given to this function. "
                    << "\n\t objects["<<i<<"].size(): " << objects[i].size()
                    << "\n\t scales["<<i<<"].size(): " << scales[i].size()
                );*/

            }
        }
#endif

        //running_stats<double> rs;
        double rs = 0;
        int count = 0;
        for (unsigned long i = 0; i < objects.size(); ++i)
        {
            for (unsigned long j = 0; j < objects[i].size(); ++j)
            {
                // Just use a scale of 1 (i.e. no scale at all) if the caller didn't supply
                // any scales.
                //const double scale = scales.size()==0 ? 1 : scales[i][j];
                const double scale = 1;

                ObjectDetection det = sp.detect(*images[i], objects[i][j]->get_rect());
/*std::cout << *objects[i][j] << std::endl;
std::cout << det << std::endl;

Mat output;
cv::cvtColor(*images[i], output, CV_GRAY2BGR);
plotFace(output, *objects[i][j], Scalar(255,0,0), 2);
plotFace(output, det, Scalar(0,0,255));
cv::imshow("Fit", output);

char key = 0;
do { key = cv::waitKey(0); } while (key != 'q');*/

                for (unsigned long k = 0; k < det.num_parts(); ++k)
                {
					Point2f gold, fit;
					fit.x = round( det.part(k).x );
					fit.y = round( det.part(k).y );
					gold.x = objects[i][j]->part(k).x;
					gold.y = objects[i][j]->part(k).y;
					//if (fit.x != gold.x || fit.y != gold.y)
std::cout << "Point[" << k << "] = (" << fit.x << ", " << fit.y << ")    Gold = (" << gold.x << ", " << gold.y << ")" << std::endl;
                    double score = mylength(fit - gold)/scale;
                    rs += score;
                    ++count;
                    //rs.add(score);
                }
            }
        }
        //return rs.mean();
        return rs / count;
    }



ShapePredictor::ShapePredictor (
	const Mat& initial_shape_,
	const std::vector<std::vector<RegressionTree> >& forests_,
	const std::vector<std::vector<Point2f > >& pixel_coordinates
) : initial_shape(initial_shape_), forests(forests_)
/*!
	requires
		- initial_shape.size()%2 == 0
		- forests.size() == pixel_coordinates.size() == the number of cascades
		- for all valid i:
			- all the index values in forests[i] are less than pixel_coordinates[i].size()
		- for all valid i and j:
			- forests[i][j].leaf_values.size() is a power of 2.
			  (i.e. we require a tree with all the levels fully filled out.
			- forests[i][j].leaf_values.size() == forests[i][j].splits.size()+1
			  (i.e. there need to be the right number of leaves given the number of splits in the tree)
!*/
{
	anchor_idx.resize(pixel_coordinates.size());
	deltas.resize(pixel_coordinates.size());
	// Each cascade uses a different set of pixels for its features.  We compute
	// their representations relative to the initial shape now and save it.
	for (unsigned long i = 0; i < pixel_coordinates.size(); ++i)
		create_shape_relative_encoding(initial_shape, pixel_coordinates[i], anchor_idx[i], deltas[i]);
}


ObjectDetection ShapePredictor::detect(
	const Mat& img,
	const Rect& rect
) const
{
	Mat current_shape;
	initial_shape.copyTo(current_shape);
//std::cout << "detect initial shape \n" << initial_shape << std::endl;
	std::vector<double> feature_pixel_values;
	for (unsigned long iter = 0; iter < forests.size(); ++iter)
	{
		extract_feature_pixel_values(img, rect, current_shape, initial_shape, anchor_idx[iter], deltas[iter], feature_pixel_values);

		/*for (int i = 0; i < feature_pixel_values.size(); ++i)
			std::cout << feature_pixel_values[i] << std::endl;*/
		// evaluate all the trees at this level of the cascade.
		for (unsigned long i = 0; i < forests[iter].size(); ++i)
			current_shape += forests[iter][i](feature_pixel_values);
	}

	// convert the current_shape into a full_object_detection
	const PointTransformAffine tform_to_img = unnormalizing_tform(rect);
	std::vector<Point2f> parts(current_shape.cols);
	for (unsigned long i = 0; i < parts.size(); ++i)
		parts[i] = tform_to_img(location(current_shape, i));
	return ObjectDetection(rect, parts);
}


}
