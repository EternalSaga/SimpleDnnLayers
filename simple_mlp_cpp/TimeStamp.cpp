#include "TimeStamp.hpp"

namespace RLVulkan {
	TimeStamp::TimeStamp()
	{
	}

	void TimeStamp::setStart()
	{
		timeStart = std::chrono::high_resolution_clock::now();
	}

	void TimeStamp::setEnd()
	{
		timeEnd = std::chrono::high_resolution_clock::now();
	}
}

