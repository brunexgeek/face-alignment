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


std::ostream &Serializable::serialize( std::ostream &out, uint16_t value )
{
	out.write((char*)(&value), sizeof(uint16_t));
	return out;
}


std::ostream &Serializable::serialize( std::ostream &out, int16_t value )
{
	serialize( out, (uint16_t) value );
	return out;
}


std::ostream &Serializable::serialize( std::ostream &out, uint8_t value )
{
	out.write((char*)(&value), sizeof(uint8_t));
	return out;
}


std::ostream &Serializable::serialize( std::ostream &out, int8_t value )
{
	serialize( out, (uint8_t) value );
	return out;
}


std::ostream &Serializable::serialize( std::ostream &out, bool value )
{
	serialize( out, (uint8_t) (value != 0) );
	return out;
}


std::ostream &Serializable::serialize( std::ostream &out, uint64_t value )
{
	out.write((char*)(&value), sizeof(uint64_t));
	return out;
}


std::ostream &Serializable::serialize( std::ostream &out, const cv::Mat& value )
{
	assert(value.isContinuous());
	
	serialize(out, (uint32_t) value.rows);
	serialize(out, (uint32_t) value.cols);
	serialize(out, (uint32_t) value.type());
	serialize(out, (uint32_t) value.elemSize());
	out.write( (char*) value.data, value.total() * value.elemSize());
	return out;
}


std::ostream &Serializable::serialize( std::ostream &out, const cv::Point2f& value )
{
	serialize(out, value.x);
	serialize(out, value.y);
	return out;
}


std::istream &Serializable::deserialize( std::istream &in, float& value )
{
	in.read((char*)(&value), sizeof(float));
	return in;
}


std::istream &Serializable::deserialize( std::istream &in, double& value )
{
	in.read((char*)(&value), sizeof(double));
	return in;
}


std::istream &Serializable::deserialize( std::istream &in, uint32_t& value )
{
	in.read((char*)(&value), sizeof(uint32_t));
	return in;
}


std::istream &Serializable::deserialize( std::istream &in, int32_t& value )
{
	deserialize( in, *((uint32_t*) &value) );
	return in;
}


std::istream &Serializable::deserialize( std::istream &in, uint16_t& value )
{
	in.read((char*)(&value), sizeof(uint16_t));
	return in;
}


std::istream &Serializable::deserialize( std::istream &in, int16_t& value )
{
	deserialize( in, *((uint16_t*) &value) );
	return in;
}


std::istream &Serializable::deserialize( std::istream &in, uint8_t& value )
{
	in.read((char*)(&value), sizeof(uint8_t));
	return in;
}


std::istream &Serializable::deserialize( std::istream &in, int8_t& value )
{
	deserialize( in, *((uint8_t*) &value) );
	return in;
}


std::istream &Serializable::deserialize( std::istream &in, bool& value )
{
	uint8_t temp;
	deserialize( in, temp );
	value = (temp != 0);
	return in;
}


std::istream &Serializable::deserialize( std::istream &in, uint64_t& value )
{
	in.read((char*)(&value), sizeof(uint64_t));
	return in;
}


std::istream &Serializable::deserialize( std::istream &in, cv::Mat& value )
{
	uint32_t rows, cols, type, elementSize;
	char *data = NULL;
	
	deserialize(in, rows);
	deserialize(in, cols);
	deserialize(in, type);
	deserialize(in, elementSize);
	data = new char[rows * cols * elementSize];
	in.read( data, rows * cols * elementSize );
	
	value = cv::Mat(rows, cols, type, data);
	
	return in;
}


std::istream &Serializable::deserialize( std::istream &in, cv::Point2f& value )
{
	deserialize(in, value.x);
	deserialize(in, value.y);
	return in;
}


} // ert
