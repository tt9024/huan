using namespace Mts::Networking;
using boost::asio::ip::tcp;


template<typename T>
inline void CTCPServer::broadcast(const T &objDataItem,
        BroadcastMessage iMsgType) {

    std::string strMessage = objDataItem.toString();

    try {

        boost::mutex::scoped_lock objScopedLock(m_MutexClients);

        for (ClientArray::iterator iter = m_Clients.begin(); iter != m_Clients.end(); ++iter) {

            if ((*iter)->isAlive() == false)
                continue;

            (*iter)->write(static_cast<unsigned int>(iMsgType));
            (*iter)->write(strMessage);
        }
    }
    catch (std::exception &e) {

        std::cerr << e.what() << std::endl;
    }
}


inline void CTCPServer::broadcast(const std::string &strMessage,
        BroadcastMessage iMsgType) {

    try {

        boost::mutex::scoped_lock objScopedLock(m_MutexClients);

        for (ClientArray::iterator iter = m_Clients.begin(); iter != m_Clients.end(); ++iter) {

            if ((*iter)->isAlive() == false)
                continue;

            (*iter)->write(static_cast<unsigned int>(iMsgType));
            (*iter)->write(strMessage);
        }
    }
    catch (std::exception &e) {

        std::cerr << e.what() << std::endl;
    }
}


inline void CTCPServer::broadcast(boost::shared_ptr <CTCPConnection> ptrConnection,
        const std::string &strDataItem,
        BroadcastMessage iMsgType) {

    try {

        ptrConnection->write(static_cast<unsigned int>(iMsgType));
        ptrConnection->write(strDataItem);
    }
    catch (std::exception &e) {

        std::cerr << e.what() << std::endl;
    }
}


