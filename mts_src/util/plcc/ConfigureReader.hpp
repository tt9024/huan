#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <map>

#include <memory>
#include <iostream>
#include <sstream>
#include <type_traits>

/* ----------------------------------
 * A configuration File Parser/Writer
 *     -- Bill Fu 2020-12
 * ----------------------------------
 *
 * Config file in the format:
 *
 * key = value \n                     (1)
 * key = { key1 = value1 \n ... } \n  (2)
 * key = [ value, ... , value ] \n    (3)
 *
 * (1): basic form, a new line is needed for each kv pair
 *      * key and value are allowed to have white-space char in the middle,
 *      * key is taken as white-space stripped string between last '\n', or '{' and '='
 *      * value is taken as white-space stripped string between '=' and '\n'
 * (2): composite form with '{' and '}'
 *      * same key/value requirements with (1)
 *      * each key/value pair in each line
 * (3): array defined within '[' and ']', with ',' as delimiter
 *      * values are taken as white-space, new-line, tab,  stripped string between
 *        previous ',' (or '[') and the next ',' (or ']')
 * (4)  comment is allowed after '#' until a new-line.
 * (5): special charactors that are used to parse the config include 
 *      '=', '{', '}', '[', ']', ',', '\n', '#'
 *      escape these charactor use '/'. Characor '/' should be escaped by itself.
 *      leading and trailing white-space can be escaped as well. white-spece in the
 *      middle doesn't need to be escaped.
 *      comment is allowed after '#' until a new-line
 * (6): Charactors in config file is assumed to be printable i.e. > 32. 
 *      In particular, special charactors [16, 23] are not allowed, they are used internally
 *      to represent the special charactors.
 * (7): Currently the configure doesn't support value/array at root level. 
 *      i.e. at root level, a value or an array without a key is illegal.
 *
 * When querying, use the '.' to navigate tree and '[]' for array elements.
 * For example:
 * Query: 'SPX.contracts[1].exch_symbol'
 * Results in looking into composit value of SPX's array of contracts element 1's value of 'exch_symbol'
 *
 * The value can be update and persisted to a same formated config file.
 *
 * Update on 2022-11-22
 * Adding supports JSON read/write by 
 * (1): optional equal char of ':',  EQCHAR
 * (2): optional key string wraps with '"'  STRWRAP
 * (3): optional ',' in between key-value pairs  KVDELIMITER
 * (4): allowing open/close brackets at start and end
 * (5): A thin object Json derived with all the necessary options
 */

#define DEFAULT_EQCHAR '='       // the "equal" char, ':' for yaml or json
#define DEFAULT_KEYWRAP ""       // the string before and after the key-value, "\"" for yaml or json
#define DEFAULT_KV_DELIMITER '\n'  // extra char needed in between kv, in addition to a newline, 
                                  // i.e. {a = b,\n c = d} NOTE it is NOT for array element delimiters

namespace utils {

class ConfigureReader {
public:
    // Value object which could hold one of three types: a map, an array or a simple value
    class Value;
    using ConfigMapType = std::map<std::string, Value>;
    class Value {
    public:
        // todo - consider move to a anonyous union, but the
        // cost of otherwise is not high, so lower priority. 
        //
        // enum {Compisit, Array, Value} tag
        // union {
        ConfigMapType _kv;
        std::vector<Value> _array;
        std::string _value;
        // }
        const ConfigureReader& _reader;  // need to get the reader's settings, i.e. native or json

        explicit
        Value(const ConfigureReader& reader) :
            _reader(reader)
            {};

        //This is invalid, non-standard c++ code Should return Value& -Werror=depreciated-copy
        //void operator=(const Value& value); // assignment of data without _reader reference

        // gets simple value or array value from current Value object,
        // and convert to the given data types.  
        // allowed types for simple values are int/long long/double and string
        // and vector of such types. 
        template<typename T> T get() const;
        template<typename T> std::vector<T> getArr() const;
        size_t arraySize() const;

        // gets composite key-value map from current Value object.
        // The key could have '.' or '[]' to suggest the composite or arrays
        // Example: asset.SPX[1].exch_symbol
        const Value* query(const std::string& key) const;

        // persists the current value into a ConfigureReadable format
        std::string toString(int indent = 2, bool array_value = false) const;
        bool operator== (const Value& val) const;

        // set the given "val" as a simple value or an array of simple values to the current Value object.
        // allowed simple value types are int/long long/double and string
        // if val is vector of these types, it makes this Value object to be an array of such type
        template<typename T> void set(const T& val);
        template<typename T> void setArr(const std::vector<T>& val);

        // it append an element to the array in the Value object
        // returns a reference of newly appended Value object to be modified.
        Value* addArrayElement();

        // adds key to the composite map of this Value object
        // if over_write is false, throws if key exists
        Value* addKey(const std::string& key, bool over_write=true);

        // removes all the settings in the Value object
        void clear();
        bool isEmpty() const;

    protected:
        // helper to set value of type T to a string for
        template<typename T>
        std::string valToString(const T& v) const;
        struct KeyItem {
            std::string _key;
            int _idx;
            KeyItem(const std::string& k, int ix): _key(k), _idx(ix) {};
        };

        // this disects a query string "key" to a vector of KeyItems
        // for example: a.b.c[2].d to 
        // [ {"a",0}, {"b",0}, {"c",0}, {"",2}, {"d",0} ]
        bool getKeyItems(const std::string& key,  std::vector<KeyItem>& ki) const;

        void assertArray() const;
        void assertMap() const;
        void assertValue() const;
    };
    // for read interface, default to be native, see also getJson()
    explicit ConfigureReader(const char* configFileName,
                             const char equal_char = DEFAULT_EQCHAR,
                             const std::string key_string_wrap = DEFAULT_KEYWRAP,
                             const char kv_delimit = DEFAULT_KV_DELIMITER);

    // create an empty m_value, for set interface
    ConfigureReader(const char equal_char = DEFAULT_EQCHAR,
                    const std::string key_string_wrap = DEFAULT_KEYWRAP,
                    const char kv_delimit = DEFAULT_KV_DELIMITER);

    // create a JSON reader/writer with config file
    static std::shared_ptr<ConfigureReader> getJson(const char* jsonFileName);

    //explicit ConfigureReader(const ConfigureReader& reader); // use default

    //void operator=(const ConfigureReader& reader);
    ~ConfigureReader();

    // clear everything and reload
    std::string reset(const std::string& configFileName = "");

    // retrieve for simple values or array
    // The key could have '.' or '[]' to suggest the composite or arrays
    // this would throw if key not found
    // Example: asset.SPX[1].exch_symbol
    template<typename T> T              get(const char* key) const;
    template<typename T> std::vector<T> getArr(const char* key) const;

    // retrieve simple value or array with default value
    template<typename T> T              get(const char* key, bool* found, const T& defaultVal) const;
    template<typename T> std::vector<T> getArr(const char* key, bool* found, const std::vector<T>& defaultVal) const;

    // retrieve a composite key-value map with the key for reading purpose
    ConfigureReader getReader(const std::string& key) const;

    // lists all keys if current Value object is a map
    // otherwise, returns empty vector
    std::vector<std::string> listKeys() const;

    // return size of array if current Value object is an array
    // otherwise, returns zero
    size_t arraySize() const;
    
    // persist the current settings to a string readable to both ConfigureReader and human
    std::string toString(size_t start_indent=0) const;

    // persist the current settings to a file
    bool toFile(const char* cfg_file_name, bool if_append = false) const;

    // Compare two config, for test purpose. The array sequence has to be the same in order to be equal
    bool operator==(const ConfigureReader& reader) const;

    // set a key to be the simple value or an array
    // the key can have '.' or '[]'
    template<typename T> void set   (const std::string& key, const T& val);
    template<typename T> void setArr(const std::string& key, const std::vector<T>& val);

    // create a key, return a value to be manipulated with Value's interface
    // in particular, when key = "", it returns the root, i.e. m_value,
    // without modifying anything yet.
    // This allows iteratively set at the Value object level, using Value's
    // set(), addElement() or addKey() interfaces
    Value* set(const std::string& key = "");

protected:
    const char EQCHAR;
    const std::string KEYWRAP;
    const char KVDELIMITER;
    std::string m_configFileName;

    // root value, key/value in m_value._kv
    Value m_value;

    // special char used for parsing
    // key and value
    enum {
        EQ = 16, // = or : \020
        OC = 17, // {      \021
        CC = 18, // }      \022
        OB = 19, // [      \023
        CB = 20, // ]      \024
        CM = 21, // ,      \025
        NL = 22, // \n     \026 needs to be restored in value
        WS = 23, // space  \027 needs to be restored in value
        DQ = 24, // "      \030 double quote
        BS = 25, // \      \031 back slash
    };

    char m_spchar_map[256];
    std::map<char, char> m_read_escape;
    std::map<char, char> m_write_escape;
    void set_escape_map();  // need to be called in construction
    char get_spchar_map(char ch) const;

    // protected constructore from a value, used in getReader()
    explicit ConfigureReader(const Value& value);

    // const helpers
    bool stringQuoted() const; // true if json format, false if native
    char quoteChar() const;
    char* readFile(const char* file_path) const;
    bool isWhiteSpace(const char ch) const;
    char* scan(char* bs, char* be, bool ignore_special_char=false) const;
    std::string toConfigString (const char* ks, const char* ke) const;
    std::string stripChars(const std::string& lines, char open_char, char close_char) const;
    std::string stripQuotedString (const std::string& key_string) const;
    bool isSpace(char c) const;
    void strip(const char*& ks, const char*& ke) const;
    const char* match(const char* s, const char* e, char match_char) const;
    const char* findValue(const char* s, const char* e, const char*&ks, const char*&ke, char end_char = NL) const;
    const char* findKey(const char* s, const char* e, const char*&ks, const char*&ke) const;
    bool parseValue(const char* s, const char* e, Value& val) const;
    bool parseKeyValue(const char* s, const char* e, ConfigMapType& kv) const;

    // Extensions for Json support 
    //
    // during scan, read "\n", call get_escape_after_slash('n'), return '\n', 
    // return 0 if \c cannot be escaped, i.e. "\a" doesn't make any sense
    // for json, allowed are 3:  ",\,n
    char get_escape_after_slash(char char_after_slash) const;

    // during output, writing a string, replace special char with '\'+char,
    // where char is returned by this function
    // input '\n', return 'n', input '"', return '"', etc
    // non-special char return 0
    char set_escape_for_special(char special_char) const;

    // convert a raw string that may have special char to
    // a properly escaped string
    std::string escapeString(const std::string& raw_string) const;

    // for json, check value to be either a quoted string, or
    // a number or "true" or "false"
    bool checkValue(const std::string& value) const;
    friend class Value;
};

inline 
std::shared_ptr<ConfigureReader> ConfigureReader::getJson(const char* jsonFileName) {
    return std::make_shared<ConfigureReader>(jsonFileName, ':', "\"", ',');
}

/*
 * Template Implementations
 */

// these would throw if conversion fails
template<typename T>
T ConfigureReader::Value::get() const {
    // gets the value of current Value object
    // allowed types for simple values are int/long long/double/bool and string
    // if T is of vector type, it gets an array of simple values with allowed types.
    //
    // this could throw if type conversion fails
    // i.e. if trying to convert a string to a number
    
    // all other types are not supported, yet...
    static_assert ((std::is_same<T, int>::value) ||
                   (std::is_same<T, long long>::value) ||
                   (std::is_same<T, double>::value)  ||
                   (std::is_same<T, bool>::value)  ||
                   (std::is_same<T, std::string>::value));
};

template<> inline int  ConfigureReader::Value::get<int>() const                { return std::stoi(_reader.stripQuotedString(_value)); };
template<> inline long long ConfigureReader::Value::get<long long>() const     { return std::stoll(_reader.stripQuotedString(_value)); };
template<> inline double ConfigureReader::Value::get<double>() const           { return std::stod(_reader.stripQuotedString(_value)); };
template<> inline bool ConfigureReader::Value::get<bool>() const             { return _reader.stripQuotedString(_value) == "true"; };
template<> inline std::string ConfigureReader::Value::get<std::string>() const { return _reader.stripQuotedString(_value); };

template<typename T>
inline std::vector<T> ConfigureReader::Value::getArr() const {
    assertArray();
    std::vector<T> vec;
    if (arraySize() > 0) {
        for (const auto& v : _array) {
            vec.push_back(v.get<T>());
        }
    }
    return vec;
}

template<typename T>
inline std::string ConfigureReader::Value::valToString(const T& val) const {
    std::ostringstream oss;
    oss << val;
    return oss.str();
}

template<typename T>
void ConfigureReader::Value::set(const T& val) {
    // set the given "val" as a simple value or an array of simple values to the current Value object.
    // allowed simple value types are int/long long/double and string

    // all other types are not supported, yet...
    static_assert ((std::is_same<T, int>::value) ||
                   (std::is_same<T, long long>::value) ||
                   (std::is_same<T, double>::value)  ||
                   (std::is_same<T, bool>::value)  ||
                   (std::is_same<T, std::string>::value));
}

template<> inline void ConfigureReader::Value::set<int>(const int& val) { clear() ; _value = valToString(val); };
template<> inline void ConfigureReader::Value::set<long long>(const long long& val) { clear() ; _value = valToString(val); };
template<> inline void ConfigureReader::Value::set<double>(const double& val) { clear() ; _value = valToString(val); };
template<> inline void ConfigureReader::Value::set<bool>(const bool& val) { clear() ; _value = (val? "true":"false"); };
template<> inline void ConfigureReader::Value::set<std::string>(const std::string& val) { clear() ; _value = _reader.KEYWRAP + val + _reader.KEYWRAP; };

template<typename T>
inline void ConfigureReader::Value::setArr(const std::vector<T>& val) {
    clear();
    for (const auto& v:val) {
        addArrayElement()->set(v);
    }
}

// getters and setters of ConfigureReader
template<typename T> 
inline T ConfigureReader::get(const char* key) const {
    // this throws if key doesn't exist
    bool found = false;
    T ret = get(key, &found, T());
    if (!found) {
        throw std::runtime_error(key + std::string(" not found!"));
    }
    return ret;
}

template<typename T> 
T ConfigureReader::get(const char* key, bool* found, const T& defaultVal) const {
    // this doesn't throw, but returns found and use default value
    bool fnd;
    if (!found) found = &fnd;

    *found = false;
    try {
        const auto* v = m_value.query(key);
        if (!v) {
            return defaultVal;
        }
        *found = true;
        return v->get<T>();
    } catch (const std::exception& e) {
        *found = false;
        return defaultVal;
    }
}

// getters and setters of ConfigureReader
template<typename T> 
std::vector<T> ConfigureReader::getArr(const char* key) const {
    // this throws if key doesn't exist
    bool found = false;
    auto ret = getArr(key, &found, std::vector<T>());
    if (!found) {
        throw std::runtime_error(key + std::string(" not found!"));
    }
    return ret;
}

template<typename T> 
std::vector<T> ConfigureReader::getArr(const char* key, bool* found, const std::vector<T>& defaultVal) const {
    // this doesn't throw, but returns found and use default value
    bool fnd;
    if (!found) found = &fnd;

    *found = false;
    try {
        const auto* v = m_value.query(key);
        if (!v) {
            return defaultVal;
        }
        *found = true;
        return v->getArr<T>();
    } catch (const std::exception& e) {
        fprintf(stderr, "got exception getting array: %s\n", e.what());
        *found = false;
        return defaultVal;
    }
}

template<typename T>
inline void ConfigureReader::set(const std::string& key, const T& val) {
    set(key)->set(val);
}

template<typename T>
inline void ConfigureReader::setArr(const std::string& key, const std::vector<T>& val) {
    set(key)->setArr(val);
}

inline
std::vector<std::string> ConfigureReader::listKeys() const {
    std::vector<std::string> vec;
    for (const auto& kv : m_value._kv) {
        vec.push_back(kv.first);
    }
    return vec;
}


inline
size_t ConfigureReader::arraySize() const {
    return m_value.arraySize();
}

inline
size_t ConfigureReader::Value::arraySize() const {
    if (_array.size() == 1) {
        const auto& v0(_array[0]);
        if (v0.isEmpty()) {
            return 0;
        }
    }
    return _array.size();
}

inline
ConfigureReader::Value* ConfigureReader::Value::addArrayElement() {
    // it append an element to the array in the Value object
    // returns a reference of newly appended Value object to be modified.
    _kv.clear();
    _value.clear();
    _array.push_back(Value(_reader));
    return &(_array[_array.size()-1]);
}

inline
void ConfigureReader::Value::assertArray() const {
    if ( (_array.size() == 0) &&
         ((_kv.size() > 0) || (_value.size() > 0))
       ) {
       throw std::runtime_error("Not an array!");
    };
}

inline
void ConfigureReader::Value::assertMap() const {
    if ( (_kv.size() == 0) &&
         ((_array.size() > 0) || (_value.size() > 0))
       ) {
       throw std::runtime_error("Not a map!");
    };
}

inline
void ConfigureReader::Value::assertValue() const {
    if ( (_value.size() == 0) &&
         ((_array.size() > 0) || (_kv.size() > 0))
       )  {
       throw std::runtime_error("Not a simple value!");
    };
}

inline
void ConfigureReader::Value::clear() {
    _value.clear();
    _array.clear();
    _kv.clear();
}

inline
bool ConfigureReader::Value::isEmpty() const {
    return _kv.size()==0 && _value.size()==0 && _array.size()==0;
}

/*
inline
ConfigureReader::Value& ConfigureReader::Value::operator=(const Value& value)
{
    _kv    = value._kv;
    _array = value._array;
    _value = value._value;
    // keep _reader reference
};
*/

inline
bool ConfigureReader::isWhiteSpace(const char ch) const {
    return ch==' ' || ch=='\t';
}

inline
bool ConfigureReader::isSpace(char c) const  {
    return (c == ' ') || (c == '\t') || (c == '\n') || (c==NL) ;
}

inline
bool ConfigureReader::toFile(const char* cfg_file_name, bool if_append) const {
    FILE* fp = fopen(cfg_file_name, (if_append?"a":"w"));
    if (!fp) return false;
    if (if_append) fprintf(fp, "\n");
    fprintf(fp, "%s\n", toString().c_str());
    fclose(fp);
    return true;
}

inline
ConfigureReader::ConfigureReader(
        const char equal_char,
        const std::string key_string_wrap,
        const char kv_delimiter) 
: EQCHAR(equal_char), KEYWRAP(key_string_wrap), KVDELIMITER(kv_delimiter), m_value(*this)
{
    if (KEYWRAP.size() > 1) {
        throw std::runtime_error("KEYWRAP size has to be less or equal to 1");
    };
    set_escape_map();
}

inline
ConfigureReader::ConfigureReader(
        const char* configFileName, 
        const char equal_char,
        const std::string key_string_wrap,
        const char kv_delimiter)
: EQCHAR(equal_char), KEYWRAP(key_string_wrap), KVDELIMITER(kv_delimiter), m_configFileName(configFileName?configFileName:""), m_value(*this)
{
    if (KEYWRAP.size() > 1) {
        throw std::runtime_error("KEYWRAP size has to be less or equal to 1");
    }
    set_escape_map();
    if (!configFileName) {
        // similar with set interface
        return;
    }
    std::string response = reset();
    if (response != "") {
        throw std::runtime_error(response);
    }
}

inline
ConfigureReader::ConfigureReader(const ConfigureReader::Value& value) :
    EQCHAR(value._reader.EQCHAR),
    KEYWRAP(value._reader.KEYWRAP),
    KVDELIMITER(value._reader.KVDELIMITER),
    m_value(value)
{
    set_escape_map();
    //m_value._reader = (const ConfigureReader) (* (const ConfigureReader*) this) ;
}

/*
inline
ConfigureReader& ConfigureReader::operator= (const ConfigureReader& reader)
{
    m_configFileName = reader.m_configFileName;

    // this copies data without _reader reference, 
    // i.e. format (nattive or json) is not changed
    m_value = reader.m_value;
}*/

inline
ConfigureReader::~ConfigureReader() { };

inline
ConfigureReader ConfigureReader::getReader(const std::string& key) const {
    const Value* v = m_value.query(key);
    if (v) {
        return ConfigureReader(*v);
    }
    fprintf(stderr, "Failed to query %s from %s\n", key.c_str(), toString().c_str());
    throw std::runtime_error("Failed to query " + key);
}

inline
ConfigureReader::Value* ConfigureReader::set(const std::string& key) {
    auto* v = m_value.addKey(key, true);
    if (!v) {
        throw std::runtime_error("key " + key + " failed to be added!");
    }
    return v;
}

inline
char ConfigureReader::get_escape_after_slash(char char_after_slash) const {
    const auto iter = m_read_escape.find(char_after_slash);
    if (iter == m_read_escape.end()) {
        return 0;
    }
    return iter->second;
}

inline
char ConfigureReader::set_escape_for_special(char special_char) const {
    const auto iter = m_write_escape.find(special_char);
    if (iter == m_read_escape.end()) {
        return 0;
    }
    return iter->second;
}

inline
std::string ConfigureReader::escapeString(const std::string& raw_string) const {
    std::string ret;
    for (size_t i=0; i<raw_string.size(); ++i) {
        char ch = raw_string[i];
        char escape_char = set_escape_for_special(ch);
        if (!escape_char) {
            ret += ch;
        } else {
            ret += '\\';
            ret += escape_char;
        }
    }
    return ret;
}

inline
bool ConfigureReader::stringQuoted() const {
    return KEYWRAP.size() > 0;
}

inline
char ConfigureReader::quoteChar() const {
    if (stringQuoted()){
        return KEYWRAP[0];
    }
    return 0;
}

inline
bool ConfigureReader::operator==(const ConfigureReader& reader) const {
    return m_value == reader.m_value;
}

}
