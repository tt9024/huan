#ifndef CSYMBOLGREATER_HEADER

#define CSYMBOLGREATER_HEADER

#include "CSymbol.h"

namespace Mts
{
	namespace Core
	{
		class CSymbolGreater
		{
		public:
			bool operator()(boost::shared_ptr<const CSymbol> ptrLHS, boost::shared_ptr<const CSymbol> ptrRHS) const {
				return ptrLHS->getSymbol().compare(ptrRHS->getSymbol()) < 0;
			}
		};
	}
}

#endif

