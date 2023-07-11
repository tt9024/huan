#include "CUserManager.h"


using namespace Mts::User;


bool CUserManager::m_bUseSQL = true;


CUserManager & CUserManager::getInstance() {

	static CUserManager theInstance;
	return theInstance;
}


void CUserManager::setModeSQL(bool bUseSQL) {

	m_bUseSQL = bUseSQL;
}


void CUserManager::initialize() {
    /*
	Mts::SQL::CMtsDB dbConn;
	dbConn.connect();

	std::vector<Mts::SQL::CSQLUser> objUsers = dbConn.readUsers();

	for (int i = 0; i != objUsers.size(); ++i) {

		Mts::User::CUser objUser(objUsers[i].getUsername(), objUsers[i].getUID(), objUsers[i].getDefaultCcys(), objUsers[i].getAccessRights());
		m_UserMap.insert(std::pair<std::string,Mts::User::CUser>(objUser.getUID(), objUser));
		m_UserList.push_back(objUser);
	}

	dbConn.disconnect();
    */
}


Mts::User::CUser CUserManager::getUser(const std::string & strUID) const {

	return m_UserMap.at(strUID);
}


bool CUserManager::isUser(const std::string & strUID) const {

	return m_UserMap.find(strUID) != m_UserMap.end();
}


CUserManager::CUserManager() {
    
	if (m_bUseSQL == true)
		initialize();
}


