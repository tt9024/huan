#ifndef CDOUBLEEQUALTO_HEADER

#define CDOUBLEEQUALTO_HEADER

#include <functional>

namespace Mts
{
	namespace Math
	{
		class CDoubleEqualTo : public std::binary_function<double, double, bool>
		{
			public:
				CDoubleEqualTo(double dEpsilon = 1e-7) : m_dEpsilon(dEpsilon) {
				}

				bool operator()(const double & dLhs, 
												const double & dRhs) const {
					return abs(dLhs - dRhs) <= m_dEpsilon;
				}

		private:
				double m_dEpsilon;
		};
	}
}

#endif


