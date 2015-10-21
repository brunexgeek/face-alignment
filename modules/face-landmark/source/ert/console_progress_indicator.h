// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CONSOLE_PROGRESS_INDiCATOR_Hh_
#define DLIB_CONSOLE_PROGRESS_INDiCATOR_Hh_

#include <ctime>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>

namespace ert
{

	/**
	 *		This object is a tool for reporting how long a task will take
	 *		to complete.
     *
	 *  	For example, consider the following bit of code:
	 *
	 *			ProgressIndicator pbar(100)
	 *			for (int i = 1; i <= 100; ++i)
	 *			{
	 *				pbar.print_status(i);
	 *				long_running_operation();
	 *			}
	 *
	 *		The above code will print a message to the console each iteration
	 *		which shows how much time is remaining until the loop terminates.
	 */
    class ProgressIndicator
    {


		public:

			explicit ProgressIndicator (
				double target_value )
			{
				reset(target_value);
			}

			void reset (
				double target_value	)
			{
				target_val = target_value;
				start_time = 0;
				first_val = 0;
				seen_first_val = false;
				last_time = 0;
			}

			double target () const
			{
				return target_val;
			}

			inline bool print (
				double current )
			{
				const time_t cur_time = std::time(0);

				// if this is the first time print_status has been called
				// then collect some information and exit.  We will print status
				// on the next call.
				if (!seen_first_val)
				{
					start_time = cur_time;
					last_time = cur_time;
					first_val = current;
					seen_first_val = true;
					return false;
				}

				if (cur_time != last_time)
				{
					last_time = cur_time;
					double delta_t = static_cast<double>(cur_time - start_time);
					double delta_val = std::abs(current - first_val);

					// don't do anything if cur is equal to first_val
					if (delta_val < std::numeric_limits<double>::epsilon())
						return false;

					double seconds = delta_t/delta_val * std::abs(target_val - current);

					int hours = seconds / 60 / 60;
					int minutes =  seconds / 60;
					seconds = seconds - ((hours * 60 * 60) + (minutes * 60));

					std::cout << "Time remaining: " <<
						std::setfill('0') << std::setw(2) << hours << ":" <<
						std::setfill('0') << std::setw(2) << minutes << ":" <<
						std::setfill('0') << std::setw(2) << (int)seconds << "               \r"<< std::flush;

					return true;
				}

				return false;
			}

		private:

			double target_val;
			time_t start_time;
			double first_val;
			double seen_first_val;
			time_t last_time;

    };


}

#endif // DLIB_CONSOLE_PROGRESS_INDiCATOR_Hh_

