#include "ProgressIndicator.hh"


namespace ert {


ProgressIndicator::ProgressIndicator(
	double iterations )
{
	reset(iterations);
}


void ProgressIndicator::reset(
	double iterations )
{
	this->iterations = iterations;
	startTime = 0;
	firstIteration = 0;
	firstCall = true;
	lastTime = 0;
	hours = minutes = seconds = 0;
	mustWait = WAIT_TIMES;
	remaining = 0;
}


bool ProgressIndicator::update(
	double current,
	bool print )
{
	const time_t now = std::time(0);

	// if it's the first call we only retrieve the the current time
	if (firstCall)
	{
		startTime = lastTime = now;
		firstIteration = current;
		firstCall = false;
		return false;
	}
	// check if elapsed some time between the current and last call
	if (now != lastTime)
	{
		lastTime = now;
		double delta_t = static_cast<double>(now - startTime);
		double delta_val = std::abs(current - firstIteration);

		if (delta_val < std::numeric_limits<double>::epsilon())
			return false;

		int time = round(delta_t/delta_val * std::abs(iterations - current));
		if (remaining == 0) remaining = time;

		if (time > remaining && mustWait == 0 || time <= remaining)
		{
			remaining = time;
			hours = remaining / (60 * 60);
			minutes =  (remaining / 60) % 60;
			seconds = remaining % 60;
			mustWait = WAIT_TIMES;
		}
		else
			mustWait--;

		if (print)
		{
			std::cout << "Time remaining: " <<
				std::setfill('0') << std::setw(2) << hours << ":" <<
				std::setfill('0') << std::setw(2) << minutes << ":" <<
				std::setfill('0') << std::setw(2) << seconds << "\r" << std::flush;
		}

		return true;
	}

	return false;
}


} // namespace ert
