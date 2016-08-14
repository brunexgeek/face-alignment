#include <ert/ShapePredictor.hh>
#include "PointAffineTransform.hh"
#include "ProgressIndicator.hh"
#include <ert/Serializable.hh>
#include "rand/rand_kernel_1.h"



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


void printRow( std::ostream& os, bool isX, const ObjectDetection& obj )
{
	int c;


	if (isX)
		os << "X = [ ";
	else
		os << "Y = [ ";

	for (c = 0; c < (int)obj.num_parts(); ++c)
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
                const double scale = scales.size()==0 ? 1 : scales[i][j];
                //const double scale = 1;

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
					gold.x = round(objects[i][j]->part(k).x);
					gold.y = round(objects[i][j]->part(k).y);
//					if (fit.x != gold.x || fit.y != gold.y)
//std::cout << "Point[" << k << "] = (" << fit.x << ", " << fit.y << ")    Gold = (" << gold.x << ", " << gold.y << ")" << std::endl;
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
	const Rect& rect,
	ShapePredictorViewer *viewer ) const
{
	Mat current_shape;
	initial_shape.copyTo(current_shape);

	const PointTransformAffine tform_to_img = unnormalizing_tform(rect);

	std::vector<double> feature_pixel_values;
	for (unsigned long iter = 0; iter < forests.size(); ++iter)
	{
		extract_feature_pixel_values(img, rect, current_shape, initial_shape, anchor_idx[iter], deltas[iter], feature_pixel_values);

		/*for (int i = 0; i < feature_pixel_values.size(); ++i)
			std::cout << feature_pixel_values[i] << std::endl;*/
		// evaluate all the trees at this level of the cascade.
		for (unsigned long i = 0; i < forests[iter].size(); ++i)
		{
			current_shape += forests[iter][i](feature_pixel_values);

			if (viewer != NULL)
			{
				std::vector<Point2f> parts(current_shape.cols);
				for (unsigned long j = 0; j < parts.size(); ++j)
					parts[j] = tform_to_img(location(current_shape, j));
				viewer->show( iter, i, ObjectDetection(rect, parts) );
			}
		}
	}

	// convert the current_shape into a full_object_detection

	std::vector<Point2f> parts(current_shape.cols);
	for (unsigned long i = 0; i < parts.size(); ++i)
		parts[i] = tform_to_img(location(current_shape, i));
	return ObjectDetection(rect, parts);
}


void ShapePredictor::serialize( std::ostream &out ) const
{
	// serialize the initial shape
	Serializable::serialize(out, initial_shape);

	// serialize the forests
	Serializable::serialize(out, forests.size());
	for (size_t i = 0; i < forests.size(); ++i)
	{
		const std::vector<RegressionTree> &current = forests[i];

		Serializable::serialize(out, current.size());
		for (size_t j = 0; j < current.size(); ++j)
			current[j].serialize(out);
	}

	// serialize the anchors
	Serializable::serialize(out, anchor_idx.size());
	for (size_t i = 0; i < anchor_idx.size(); ++i)
	{
		const std::vector<unsigned long> &current = anchor_idx[i];

		Serializable::serialize(out, current.size());
		for (size_t j = 0; j < current.size(); ++j)
			Serializable::serialize(out, current[j]);
	}

	// serialize the deltas
	Serializable::serialize(out, deltas.size());
	for (size_t i = 0; i < deltas.size(); ++i)
	{
		const std::vector<Point2f> &current = deltas[i];

		Serializable::serialize(out, current.size());
		for (size_t j = 0; j < current.size(); ++j)
			Serializable::serialize(out, current[j]);
	}
}


void ShapePredictor::deserialize( std::istream &in )
{
	// deserialize the initial shape
	Serializable::deserialize(in, initial_shape);

	// deserialize the forests
	size_t entries;
	Serializable::deserialize(in, entries);
	forests.resize(entries);
	for (size_t i = 0; i < entries; ++i)
	{
		size_t entries;
		Serializable::deserialize(in, entries);

		std::vector<RegressionTree> &current = forests[i];
		current.resize(entries);

		for (size_t j = 0; j < current.size(); ++j)
			current[j].deserialize(in);
	}

	// deserialize the anchors
	Serializable::deserialize(in, entries);
	anchor_idx.resize(entries);
	for (size_t i = 0; i < entries; ++i)
	{
		size_t entries;
		Serializable::deserialize(in, entries);

		std::vector<unsigned long> &current = anchor_idx[i];
		current.resize(entries);

		for (size_t j = 0; j < current.size(); ++j)
			Serializable::deserialize(in, current[j]);
	}

	// serialize the deltas
	Serializable::deserialize(in, entries);
	deltas.resize(entries);
	for (size_t i = 0; i < entries; ++i)
	{
		size_t entries;
		Serializable::deserialize(in, entries);

		std::vector<Point2f> &current = deltas[i];
		current.resize(entries);

		for (size_t j = 0; j < current.size(); ++j)
			Serializable::deserialize(in, current[j]);
	}

}


}
