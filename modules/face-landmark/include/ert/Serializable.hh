#ifndef FA_SERIALIZABLE_H
#define FA_SERIALIZABLE_H


#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdint.h>


namespace ert {


class Serializable
{

	public:
		static std::ostream &serialize( std::ostream &out, float value );

		static std::ostream &serialize( std::ostream &out, double value );

		static std::ostream &serialize( std::ostream &out, uint32_t value );

		static std::ostream &serialize( std::ostream &out, int32_t value );

		static std::ostream &serialize( std::ostream &out, uint16_t value );

		static std::ostream &serialize( std::ostream &out, int16_t value );

		static std::ostream &serialize( std::ostream &out, uint8_t value );

		static std::ostream &serialize( std::ostream &out, int8_t value );

		static std::ostream &serialize( std::ostream &out, bool value );
		
		static std::ostream &serialize( std::ostream &out, uint64_t value );
		
		static std::ostream &serialize( std::ostream &out, const cv::Mat& value );
		
		static std::ostream &serialize( std::ostream &out, const cv::Point2f& value );

		static std::istream &deserialize( std::istream &in, float& value );

		static std::istream &deserialize( std::istream &in, double& value );

		static std::istream &deserialize( std::istream &in, uint32_t& value );

		static std::istream &deserialize( std::istream &in, int32_t& value );

		static std::istream &deserialize( std::istream &in, uint16_t& value );

		static std::istream &deserialize( std::istream &in, int16_t& value );

		static std::istream &deserialize( std::istream &in, uint8_t& value );

		static std::istream &deserialize( std::istream &in, int8_t& value );

		static std::istream &deserialize( std::istream &in, bool& value );
		
		static std::istream &deserialize( std::istream &in, uint64_t& value );
		
		static std::istream &deserialize( std::istream &in, cv::Mat& value );
		
		static std::istream &deserialize( std::istream &in, cv::Point2f& value );

		virtual void serialize( std::ostream &out ) const = 0;
		
		virtual void deserialize( std::istream &in ) = 0;

};


}


#endif // FA_SERIALIZABLE_H
