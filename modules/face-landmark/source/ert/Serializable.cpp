#include "Serializable.hh"


namespace ert {


std::ostream &Serializable::serialize( std::ostream &out, float value )
{
	out.write((char*)(&value), sizeof(float));
	return out;
}


std::ostream &Serializable::serialize( std::ostream &out, double value )
{
	out.write((char*)(&value), sizeof(double));
	return out;
}


std::ostream &Serializable::serialize( std::ostream &out, uint32_t value )
{
	out.write((char*)(&value), sizeof(uint32_t));
	return out;
}


std::ostream &Serializable::serialize( std::ostream &out, int32_t value )
{
	serialize( out, (uint32_t) value );
	return out;
}


std::ostream &serialize( std::ostream &out, uint16_t value )
{
	out.write((char*)(&value), sizeof(uint16_t));
	return out;
}


std::ostream &serialize( std::ostream &out, int16_t value )
{
	serialize( out, (uint16_t) value );
	return out;
}


std::ostream &serialize( std::ostream &out, uint8_t value )
{
	out.write((char*)(&value), sizeof(uint8_t));
	return out;
}


std::ostream &serialize( std::ostream &out, int8_t value )
{
	serialize( out, (uint8_t) value );
	return out;
}


std::ostream &serialize( std::ostream &out, bool value )
{
	serialize( out, (uint8_t) (value != 0) );
	return out;
}


} // ert
