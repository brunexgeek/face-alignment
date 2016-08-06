#ifndef FA_SAMPLE_LIST_HH
#define FA_SAMPLE_LIST_HH


#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ObjectDetection.hh"


namespace ert {


class SampleLoader
{

	public:
		SampleLoader()
		{
			// nothing to do
		}

		virtual ~SampleLoader()
		{
			// nothing to do
		}

		virtual void load(
			const std::string &imageFileName,
			cv::Mat &image,
			ObjectDetection &annot ) = 0;

};


class BasicSampleLoader : public SampleLoader
{

	public:
		BasicSampleLoader();

		~BasicSampleLoader();

		void load(
			const std::string &imageFileName,
			cv::Mat &image,
			ObjectDetection &annot );

};


// ScriptFile?
class SampleList
{

	public:
		SampleList(
			const std::string &fileName,
			SampleLoader *loader = NULL );

		~SampleList();

		const std::string &getImageFileName(
			uint32_t index ) const;

		const cv::Mat &getImage(
			uint32_t index ) const;

		const ObjectDetection &getAnnotation(
			uint32_t index ) const;

		const std::string getFileName(
			uint32_t index,
			const std::string &extension ) const;

		const std::vector<cv::Mat*> &getImages() const;

		const std::vector<std::vector<ObjectDetection*> > &getAnnotations()  const;

		static const std::string changeExtension(
			const std::string &fileName,
			const std::string &extension );

	private:

		std::vector<cv::Mat*> images;

		std::vector<std::vector<ObjectDetection*> > annotations;

		std::vector<std::string> imageFileNames;

};


} // namespace ert

#endif // FA_SAMPLE_LIST_HH
