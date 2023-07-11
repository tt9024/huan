#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include "CXMLParser.h"
#include "CApplicationLog.h"


using namespace Mts::XML;


CXMLParser::CXMLParser() {

}


boost::property_tree::ptree CXMLParser::parseXMLFile(const std::string & strXMLFile) const
{
	try {

		boost::property_tree::ptree objPTree;
		boost::property_tree::xml_parser::read_xml(strXMLFile, objPTree, boost::property_tree::xml_parser::trim_whitespace);
		return objPTree;
	}
	catch (std::exception & e) {
        AppError("Parsing XML file %s: %s", strXMLFile.c_str(), e.what());
		throw;
	}
}


