
#ifndef FA_LANDMARK_ERT_PROGRESS_HH
#define FA_LANDMARK_ERT_PROGRESS_HH

#include <ctime>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>

namespace ert
{


/**
 * Class to display how long a task will take to complete.
 *
 *     ProgressIndicator pbar(100)
 *	   for (int i = 1; i <= 100; ++i)
 *	   {
 *	       pbar.print_status(i);
 *         long_running_operation();
 *     }
 *
 * At each iteration, the ProgressIndicator class will show how
 * how much time is remaining until the loop terminates.
 */
class ProgressIndicator
{
	public:
		explicit ProgressIndicator(
			double iterations );

		void reset(
			double iterations );

		bool update(
			double current,
			bool print = false );

	private:
		double iterations;
		time_t startTime;
		bool firstCall;
		double firstIteration;
		time_t lastTime;
		int remaining;
		int hours, minutes, seconds;
};


}

#endif // FA_LANDMARK_ERT_PROGRESS_HH

