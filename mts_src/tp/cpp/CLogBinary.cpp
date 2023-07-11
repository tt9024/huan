#include <iostream>
#include "CLogBinary.h"
#include "CApplicationLog.h"


using namespace Mts::Log;


CLogBinary::CLogBinary() {

}


CLogBinary::~CLogBinary() {

}


void CLogBinary::openLog(const std::string & strFileName) {

	try {
		 m_ptrFile = fopen(strFileName.c_str(), "a+b");
	}
	catch (std::exception & e) {

		AppLogError(e.what());
		throw;
	}
}


void CLogBinary::openLogReadOnly(const std::string & strFileName) {

	try {
		 m_ptrFile = fopen(strFileName.c_str(), "rb");
	}
	catch (std::exception & e) {

		AppLogError(e.what());
		throw;
	}
}


void CLogBinary::closeLog() {

	fclose(m_ptrFile);
}


long CLogBinary::getSize() const {

	fseek(m_ptrFile, 0, SEEK_END);
	return ftell(m_ptrFile);
}

