#ifndef FA_SERIALIZABLE_H
#define FA_SERIALIZABLE_H


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

	private:
		virtual void serialize( std::ostream &out ) = 0;
		virtual void deserialize( std::istream &in ) = 0;

};


}


#endif // FA_SERIALIZABLE_H
