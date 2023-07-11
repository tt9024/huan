#ifndef CSTATISTICS_HEADER


#define CSTATISTICS_HEADER


#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>


namespace Mts
{
	namespace Stats
	{
		class CStatistics
		{
			public:
				CStatistics() = default;

				using E = double;
				template <typename IT>
				E mean(IT begin, IT end) {

					auto N = std::distance(begin, end);
					E average = std::accumulate(begin, end, E()) / N;

					return average;
				}

				template <typename IT>
				E stdev(IT begin, IT end) {

					auto N = std::distance(begin, end);
					E average = std::accumulate(begin, end, E()) / N;

					auto sum_term = [average](E init, E value)-> E{ return init + (value - average)*(value - average); };
					E variance = std::accumulate(begin,  end, E(), sum_term);

					return std::sqrt(variance * 1.0 / (N - 1));
				}
		};
	}
}

#endif
