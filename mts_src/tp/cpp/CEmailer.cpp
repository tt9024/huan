#define WIN32_LEAN_AND_MEAN
#define CRLF "\r\n"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>

#ifndef LINUX_COMPILATION
#include <windows.h>
#include <winsock2.h>
#endif

#include "CEmailer.h"
#include "CApplicationLog.h"
#include "CConfig.h"
#include "CDateTime.h"
#include "CStringTokenizer.h"
#include "CMtsException.h"
#include <boost/thread/thread.hpp>


const int VERSION_MAJOR = 1;
const int VERSION_MINOR = 1;


using namespace Mts::Mail;


void CEmailer::check(int iStatus, char *szFunction) const {

#ifndef LINUX_COMPILATION
    if((iStatus != SOCKET_ERROR) && (iStatus))
      return;

      throw Mts::Exception::CMtsException("Error: Cannot SMTP transmission error");
#endif
}


bool CEmailer::sendEmailStd(const std::string &strToEmailAddr,
        const std::string &strSubject,
        const std::string &strBody) const {

    if (Mts::Core::CConfig::getInstance().isEmailEnabled() == false)
        return true;

    Mts::Core::CStringTokenizer objSplitter;

    std::vector<std::string> msg;
    msg.push_back(strBody);

    std::vector<std::string> strEmail = objSplitter.splitString(strToEmailAddr, ";");

    for (size_t i = 0; i != strEmail.size(); ++i) {

        boost::shared_ptr<boost::thread> ptrThread = boost::shared_ptr<boost::thread>(new boost::thread(
                boost::bind(&CEmailer::sendEmail, this, Mts::Core::CConfig::getInstance().getSMTPServer(), strEmail[i],
                            Mts::Core::CConfig::getInstance().getFromEmailAddr(), strSubject, msg)));
    }

    return true;
}


bool CEmailer::sendEmail(const std::string &strToEmailAddr,
        const std::string &strSubject,
        const std::string &strMsgLine) const {

    if (Mts::Core::CConfig::getInstance().isEmailEnabled() == false)
        return true;

    char szBuffer[1024];
    sprintf(szBuffer, "%s: %s", Mts::Core::CDateTime::now().toStringFull().c_str(), strMsgLine.c_str());

    std::vector<std::string> msg;
    msg.push_back(szBuffer);

    msg.push_back("");
    msg.push_back("**** Please contact Eugene Morris +1 917 536 2507 (primary) +1 212 362 5985 (secondary) ****");

    return sendEmail(Mts::Core::CConfig::getInstance().getSMTPServer(), strToEmailAddr,
                     Mts::Core::CConfig::getInstance().getFromEmailAddr(), strSubject, msg);
}


bool CEmailer::sendEmail(const std::string &strSubject,
        const std::string &strMsgLine) const {

    if (Mts::Core::CConfig::getInstance().isEmailEnabled() == false)
        return true;

    char szBuffer[1024];
    sprintf(szBuffer, "%s: %s", Mts::Core::CDateTime::now().toStringFull().c_str(), strMsgLine.c_str());

    std::vector<std::string> msg;
    msg.push_back(szBuffer);

    msg.push_back("");
    msg.push_back("**** Please contact Eugene Morris +1 917 536 2507 (primary) +1 212 362 5985 (secondary) ****");

    return sendEmail(Mts::Core::CConfig::getInstance().getSMTPServer(),
                     Mts::Core::CConfig::getInstance().getToEmailAddr(),
                     Mts::Core::CConfig::getInstance().getFromEmailAddr(), strSubject, msg);
}


bool CEmailer::sendEmail(const std::string &strSubject,
        const std::vector<std::string> &strMsgLine) const {

    if (Mts::Core::CConfig::getInstance().isEmailEnabled() == false)
        return true;

    return sendEmail(Mts::Core::CConfig::getInstance().getSMTPServer(),
                     Mts::Core::CConfig::getInstance().getToEmailAddr(),
                     Mts::Core::CConfig::getInstance().getFromEmailAddr(), strSubject, strMsgLine);
}


bool CEmailer::sendEmail(const std::string &strSmtpServerName,
        const std::string &strToAddr,
        const std::string &strFromAddr,
        const std::string &strSubject,
        const std::vector<std::string> &strMsgLine) const {

    if (Mts::Core::CConfig::getInstance().isEmailEnabled() == false)
        return true;

#ifdef LINUX_COMPILATION

    FILE *ptrMailPipe = popen("/usr/lib/sendmail -t", "w");

    if (ptrMailPipe != NULL) {

        fprintf(ptrMailPipe, "To: %s\n", strToAddr.c_str());
        fprintf(ptrMailPipe, "From: %s\n", strFromAddr.c_str());
        fprintf(ptrMailPipe, "Subject: %s\n\n", strSubject.c_str());

        for (size_t i = 0; i != strMsgLine.size(); ++i) {

            fwrite(strMsgLine[i].c_str(), 1, strlen(strMsgLine[i].c_str()), ptrMailPipe);
            fwrite(".\n", 1, 2, ptrMailPipe);
        }

        pclose(ptrMailPipe);

        return true;
    } else {

        return false;
    }

#else

#define	_WINSOCK_DEPRECATED_NO_WARNINGS
    try {

        int         iProtocolPort        = 0;
        char        szSmtpServerName[64] = "";
        char        szToAddr[64]         = "";
        char        szFromAddr[64]       = "";
        char        szBuffer[4096]			 = "";
        char        szMsgLine[512000]    = "";
        SOCKET      hServer;
        WSADATA     WSData;
        LPHOSTENT   lpHostEntry;
        LPSERVENT   lpServEntry;
        SOCKADDR_IN SockAddr;


        // Load command-line args
        lstrcpy(szSmtpServerName, strSmtpServerName.c_str());
        lstrcpy(szToAddr, strToAddr.c_str());
        lstrcpy(szFromAddr, strFromAddr.c_str());


        // Attempt to intialize WinSock (1.1 or later)
        if(WSAStartup(MAKEWORD(VERSION_MAJOR, VERSION_MINOR), &WSData)) {
            std::cout << "Cannot find Winsock v" << VERSION_MAJOR << "." << VERSION_MINOR << " or later!" << std::endl;

            return false;
        }

        // Lookup email server's IP address.
        lpHostEntry = gethostbyname(szSmtpServerName);

        if(!lpHostEntry) {
            std::cout << "Cannot find SMTP mail server " << szSmtpServerName << std::endl;

            return false;
        }

        // Create a TCP/IP socket, no specific protocol
        hServer = socket(PF_INET, SOCK_STREAM, 0);

        if(hServer == INVALID_SOCKET) {
            std::cout << "Cannot open mail server socket" << std::endl;

            return false;
        }

        // Get the mail service port
        lpServEntry = getservbyname("mail", 0);

        // Use the SMTP default port if no other port is specified
        if(!lpServEntry)
            iProtocolPort = htons(IPPORT_SMTP);
        else
            iProtocolPort = lpServEntry->s_port;

        // Setup a Socket Address structure
        SockAddr.sin_family = AF_INET;
        SockAddr.sin_port   = iProtocolPort;
        SockAddr.sin_addr   = *((LPIN_ADDR)*lpHostEntry->h_addr_list);

        // Connect the Socket
        if(connect(hServer, (PSOCKADDR) &SockAddr, sizeof(SockAddr))) {
            std::cout << "Error connecting to Server socket" << std::endl;

            return false;
        }

        // Receive initial response from SMTP server
        check(recv(hServer, szBuffer, sizeof(szBuffer), 0), (char*)"recv() Reply");

        // Send HELO server.com
        sprintf(szMsgLine, "HELO %s%s", szSmtpServerName, CRLF);
        check(send(hServer, szMsgLine, strlen(szMsgLine), 0), (char*)"send() HELO");
        check(recv(hServer, szBuffer, sizeof(szBuffer), 0), (char*)"recv() HELO");

        // Send MAIL FROM: <sender@mydomain.com>
        sprintf(szMsgLine, "MAIL FROM:<%s>%s", szFromAddr, CRLF);
        check(send(hServer, szMsgLine, strlen(szMsgLine), 0), (char*)"send() MAIL FROM");
        check(recv(hServer, szBuffer, sizeof(szBuffer), 0), (char*)"recv() MAIL FROM");

        // Send RCPT TO: <receiver@domain.com>
        sprintf(szMsgLine, "RCPT TO:<%s>%s", szToAddr, CRLF);
        check(send(hServer, szMsgLine, strlen(szMsgLine), 0), (char*)"send() RCPT TO");
        check(recv(hServer, szBuffer, sizeof(szBuffer), 0), (char*)"recv() RCPT TO");

        // Send DATA
        sprintf(szMsgLine, "DATA%s", CRLF);
        check(send(hServer, szMsgLine, strlen(szMsgLine), 0), (char*)"send() DATA");
        check(recv(hServer, szBuffer, sizeof(szBuffer), 0), (char*)"recv() DATA");

        sprintf(szMsgLine, "From: %s%s", szFromAddr, CRLF);
        sprintf(szMsgLine, "%sTo: %s%s", szMsgLine, szToAddr, CRLF);

        sprintf(szMsgLine, "%sMime-Version: 1.0;%s", szMsgLine, CRLF);
        sprintf(szMsgLine, "%sContent-Type: text/html;%s charset=\"ISO-8859-1\";%s", szMsgLine, CRLF, CRLF);
        sprintf(szMsgLine, "%sContent-Transfer-Encoding: 7bit;%s", szMsgLine, CRLF);

        sprintf(szMsgLine, "%sSubject: %s %s %s", szMsgLine, strSubject.c_str(), Mts::Core::CDateTime::now().toStringFull().c_str(), CRLF);
        sprintf(szMsgLine, "%s%s<html><body>", szMsgLine, CRLF);

        for (size_t i = 0; i != strMsgLine.size(); ++i) {

            sprintf(szMsgLine, "%s<font face=\"courier new\"><pre>%s</pre></font>%s", szMsgLine, strMsgLine[i].c_str(), CRLF);
        }

        sprintf(szMsgLine, "%s</body></html>%s", szMsgLine, CRLF);

        check(send(hServer, szMsgLine, strlen(szMsgLine), 0), (char*)"send() message-line");

        // Send blank line and a period
        sprintf(szMsgLine, "%s.%s", CRLF, CRLF);
        check(send(hServer, szMsgLine, strlen(szMsgLine), 0), (char*)"send() end-message");
        check(recv(hServer, szBuffer, sizeof(szBuffer), 0), (char*)"recv() end-message");

        // Send QUIT
        sprintf(szMsgLine, "QUIT%s", CRLF);
        check(send(hServer, szMsgLine, strlen(szMsgLine), 0), (char*)"send() QUIT");
        check(recv(hServer, szBuffer, sizeof(szBuffer), 0), (char*)"recv() QUIT");

        // Close server socket and prepare to exit.
        closesocket(hServer);

        WSACleanup();

        return true;
    }
    catch(const std::exception &e) {
        AppLogError(std::string("Error occurred trying to send email via SMTP: ") + std::string(e.what()));
        return false;
    }
#endif
}


