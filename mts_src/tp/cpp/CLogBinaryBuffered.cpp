#include "CLogBinaryBuffered.h"
#include "CConfig.h"
#include "CMtsException.h"
#include "CApplicationLog.h"


using namespace Mts::Log;


CLogBinaryBuffered::CLogBinaryBuffered()
: m_BufferA(Mts::Core::CConfig::getInstance().getLogBufferSize()),
	m_BufferB(Mts::Core::CConfig::getInstance().getLogBufferSize()) {

	m_iLogIntervalMillis = Mts::Core::CConfig::getInstance().getLogIntervalSec() * 1000;
	m_ptrCurrentBuffer = &m_BufferA;
}


CLogBinaryBuffered::CLogBinaryBuffered(const std::string &	strLogFile,
																			 unsigned int					iBufferSize,
																			 unsigned int					iLogIntervalMillis)
: m_strLogFile(strLogFile),
	m_BufferA(iBufferSize),
	m_BufferB(iBufferSize) {

	m_iLogIntervalMillis = iLogIntervalMillis * 1000;
	m_ptrCurrentBuffer = &m_BufferA;
}


void CLogBinaryBuffered::setLogFile(const std::string & strLogFile) {

	try {
		m_strLogFile = strLogFile;
		m_ptrFile = fopen(m_strLogFile.c_str(), "a+b");
	}
	catch (std::exception & e) {

		AppLogError(e.what());
		throw;
	}
}


void CLogBinaryBuffered::run() {

	m_ptrThread = boost::shared_ptr<boost::thread>(new boost::thread(boost::ref(*this)));
}


void CLogBinaryBuffered::operator()() {

	while (!isStopped()) {

		Buffer * ptrBufferToFlush = NULL;

		{
			boost::mutex::scoped_lock lock(m_Mutex);
			ptrBufferToFlush = m_ptrCurrentBuffer;
			m_ptrCurrentBuffer = (m_ptrCurrentBuffer == &m_BufferA) ? &m_BufferB : &m_BufferA;
		}

		flushBuffer(*ptrBufferToFlush);

		boost::this_thread::sleep(boost::posix_time::milliseconds(m_iLogIntervalMillis));
	}
}


void CLogBinaryBuffered::flushBuffer(Buffer & objBuffer) {

	while (objBuffer.getNumItems() != 0) {

		unsigned int iNumBytes = 0;
		objBuffer.pop(reinterpret_cast<char *>(&iNumBytes), sizeof(iNumBytes), false);

		if (iNumBytes > BUFFER_SIZE) {

			char szError[255];
			sprintf(szError, "binary buffer too small, item size = %u, buffer size = %d", iNumBytes, BUFFER_SIZE);
			throw Mts::Exception::CMtsException(szError);
		}

		char szBuffer[BUFFER_SIZE];
		objBuffer.pop(szBuffer, iNumBytes, false);

		fwrite(reinterpret_cast<const void *>(&iNumBytes), sizeof(iNumBytes), 1, m_ptrFile);
		fwrite(reinterpret_cast<const void *>(szBuffer), iNumBytes, 1, m_ptrFile);
	}
}


bool CLogBinaryBuffered::startLogging() {

	if (m_ptrFile != NULL) {

		m_bStopped = false;
		run();

		return true;
	}
	else
		return false;
}


bool CLogBinaryBuffered::isStopped() const {

	return m_bStopped;
}


bool CLogBinaryBuffered::stopLogging() {

	try {
		m_bStopped = true;
		m_ptrThread->join();
		flushBuffer(*m_ptrCurrentBuffer);
		fclose(m_ptrFile);

		return true;
	}
	catch (std::exception & e) {

		AppLogError(e.what());
		return false;
	}
}


