#pragma once
#include <chrono>
#include <ratio>
#include <exception>

//Example

//TimeStamp ts;
//ts.setStart();
//for (size_t i = 0; i < 10000000; i++)
//{
//	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 5, 5, 5, 1.0f, a.get(), 5, b.get(), 5, 0.0f, c.get(), 5);
//}
//ts.setEnd();
//std::cout << ts.getElapsedTime<std::chrono::milliseconds>() << std::endl;
//


namespace RLVulkan {
	class TimeStamp
	{
		std::chrono::time_point<std::chrono::high_resolution_clock> timeStart;
		std::chrono::time_point<std::chrono::high_resolution_clock> timeEnd;

	public:
		constexpr static uint64_t FenceTimeout{ 100000000 };
		explicit TimeStamp();
		~TimeStamp() = default;
		void setStart();
		void setEnd();

		// timeUnit is std::chrono::microseconds 
		// std::chrono::milliseconds std::chrono::seconds etc
		template<typename timeUnit>
		auto getElapsedTime() -> decltype(std::chrono::duration_cast<timeUnit>(timeEnd - timeStart).count());
	};

	template<typename timeUnit>
	inline auto TimeStamp::getElapsedTime() -> decltype(std::chrono::duration_cast<timeUnit>(timeEnd - timeStart).count())
	{
		return std::chrono::duration_cast<timeUnit>(timeEnd - timeStart).count();
	}
}


