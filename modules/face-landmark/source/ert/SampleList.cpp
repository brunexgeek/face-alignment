#include "SampleList.hh"
#include <fstream>

namespace ert {


BasicSampleLoader::BasicSampleLoader()
{
}

BasicSampleLoader::~BasicSampleLoader()
{
}


void BasicSampleLoader::load(
	const std::string &imageFileName,
	cv::Mat &image,
	ObjectDetection &annot )
{
	// load the image
#if (0)
	cv::cvtColor(imread(line), image, CV_BGR2GRAY);
#else
	cv::Mat temp[3];
	cv::split(imread(imageFileName), temp);
	temp[0].convertTo(temp[0], CV_32F);
	temp[1].convertTo(temp[1], CV_32F);
	temp[2].convertTo(temp[2], CV_32F);
	image = temp[0] + temp[1] + temp[2];
	image /= 3.0;
	image.convertTo(image, CV_8U);
#endif
	// load the annotations
	std::string pointsFile = SampleList::changeExtension(imageFileName, "pts");
	try
	{
		annot.load(pointsFile);
	} catch (...)
	{
		std::cout << "   Ignoring file " << pointsFile << std::endl;
	}
}

SampleList::SampleList(
	const std::string &fileName,
	SampleLoader *loader )
{
	char line[256];
	int imageCount = 0;
	BasicSampleLoader defaultLoader;

	if (loader == NULL) loader = &defaultLoader;

	// open script file
	std::ifstream list(fileName.c_str());
	// read the amount of files
	list.getline(line, sizeof(line));
	if (sscanf(line, "%d", &imageCount) != 1)
		throw 1;

	images.resize(imageCount);
	annotations.resize(imageCount);
	imageFileNames.resize(imageCount);

	for (int i = 0; i < imageCount; ++i)
	{
		list.getline(line, sizeof(line) - 5);

		std::cout << "Loading " << i << " of " << imageCount << ": " << line << std::endl;

		try
		{
			// call the loader to load the image and annotations
			cv::Mat *image = new cv::Mat();
			ObjectDetection *annot = new ObjectDetection();
			loader->load(line, *image, *annot);
			// fill the internal lists
			imageFileNames[i] = line;
			images[i] = image;
			std::vector<ObjectDetection*> temp;
			temp.push_back(annot);
			annotations[i] = temp;
		} catch (...)
		{
			std::cout << "Ops" << std::endl;
		}
	}
}


SampleList::~SampleList()
{
	// TODO: free memory here
}


const std::string SampleList::changeExtension(
	const std::string &fileName,
	const std::string &extension )
{
	std::string temp = fileName;

	size_t pos = temp.find_last_of(".");
	if (pos != std::string::npos)
	{
		temp.erase(pos+1);
	}

	temp += extension;
	return temp;
}


const std::string& SampleList::getImageFileName(
	uint32_t index ) const
{
	return imageFileNames[index];
}


const cv::Mat &SampleList::getImage(
	uint32_t index ) const
{
	return *images[index];
}


const ObjectDetection &SampleList::getAnnotation(
	uint32_t index ) const
{
	return *annotations[index][0];
}

const std::vector<cv::Mat*>& SampleList::getImages() const
{
	return images;
}

const std::vector<std::vector<ObjectDetection*> > &SampleList::getAnnotations() const
{
	return annotations;
}


const std::string SampleList::getFileName(
	uint32_t index,
	const std::string &extension ) const
{
	return SampleList::changeExtension(imageFileNames[index], extension);
}


} // namespace ert

