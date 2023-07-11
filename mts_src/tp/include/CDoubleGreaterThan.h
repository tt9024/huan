#ifndef CDOUBLEGREATERTHAN_HEADER

#define CDOUBLEGREATERTHAN_HEADER

#include <functional>

namespace Mts
{
	namespace Math
	{
		class CDoubleGreaterThan : public std::binary_function<double, double, bool>
		{
			public:
				CDoubleGreaterThan(double dEpsilon = 1e-7) : m_dEpsilon(dEpsilon) {
				}

				bool operator()(const double & dLhs, 
												const double & dRhs) const {
					return (abs(dLhs - dRhs) > m_dEpsilon) && (dLhs > dRhs);
				}

		private:
				double m_dEpsilon;
		};
	}
}

#endif


