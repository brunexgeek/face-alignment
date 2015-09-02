#ifndef FA_SERIALIZABLE_H
#define FA_SERIALIZABLE_H


#include <iostream>


namespace ert {


class Serializable
{

	public:
		static void serialize( float value, std::ostream &out )
		{
			out.write((char*)(&value), sizeof(float));
		}


		static void serialize( double value, std::ostream &out )
		{
			out.write((char*)(&value), sizeof(double));
		}


		static void serialize( uint32_t value, std::ostream &out )
		{
			out.write((char*)(&value), sizeof(uint32_t));
		}


		static void serialize( int32_t value, std::ostream &out )
		{
			serialize(value, out);
		}

	private:
		virtual void serialize( std::ostream &out ) = 0;
		virtual void deserialize( std::istream &in ) = 0;

};


}


#endif // FA_SERIALIZABLE_H
