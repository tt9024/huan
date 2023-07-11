#include "ConfigureReader.hpp"
#include <string.h>

namespace utils {

char* ConfigureReader::readFile(const char* file_path) const {
    // read the file into a string.
    FILE* fp = fopen(file_path, "r");
    if (!fp) {
        fprintf(stderr, "ConfigureReader failed to open %s\n", file_path);
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    size_t sz = ftell(fp);

    /* allow empty file for config file builder
    if (sz < 3) {
        fprintf(stderr, "ConfigureReader file too small %s\n", file_path);
        fclose(fp);
        return nullptr;
    }
    */

    char* buf = (char*) malloc(sz+2);
    fseek(fp, 0, SEEK_SET);
    fread(buf, 1, sz, fp);
    buf[sz] = '\n';
    buf[sz+1] = 0;
    fclose(fp);
    return buf;
}

char* ConfigureReader::scan(char* bs, char* be, bool ignore_special_char) const {
    // 1. remove comments
    // 2. remove lines with only white spaces
    // 3. replace speical charactors
    // 4. remove escape slash for special charactors
    //
    // input:
    //    bs: the first charactor to be scanned
    //    be: the last charactor to be scanned (not including terminating 0)
    // return:
    //    pe - the last charactor of modified string.  The modification is in-place,
    //    therefore [bs, pe] (inclusive) is the string to be used later.
    //    Note no terminating 0 is added, pe is the last charactor, not the terminating 0.
    //
    // state: 0: new line
    //        1: comments
    //        2: escape
    //        3: normal
    int state = 0;
    char* const ps = bs; // ps points to the start of the string, to be modified in *p
    char* p = bs;
    while (bs <= be) {
        if (((int)*bs < 32) && (*bs != '\n') && (*bs != '\t')) {
            if (((int)*bs != 13) &&  !ignore_special_char) {
                fprintf(stderr, "unrecognized char with ascii code %d\n", (int) *bs);
                return nullptr;
            }
            ++bs;
            continue;
        }

        if (state == 1) {
            // comments, remove until a new line
            if (*bs == '\n') {
                state = 3;
                --bs; // need this new line
            }
            ++bs;
            continue;
        }
        if (state == 2) {
            // escape, have to be a special char or '\'
            char ch = get_escape_after_slash(*bs++);
            if ( ch != 0) {
                *p++ = ch;
                state = 3;
                continue;
            }
            fprintf(stderr, "error found for escape of (%c), (forgot to escape \'\\\'?) %s\n", *(bs-1), std::string(ps, bs-ps+1).c_str());
            return nullptr;
        }
        if (state == 0) {
            // remove white space
            if (isWhiteSpace(*bs) || (*bs == '\n')) {
                // skip leading while spaces or empty lines
                ++bs;
                continue;
            }
            state = 3;
            // fall through
        }

        // state is 3, normal
        switch (*bs) {
            case '#' :
            {
                state = 1;
                break;
            }
            case '\\':
            {
                state = 2;
                break;
            }
            case '\n':
            {
                state = 0; [[fallthrough]];
                // this falls down
            }
            default :
            {
                if (stringQuoted() && (*bs == quoteChar()) ) {
                    // in quote, take in until quote
                    *p++ = *bs++;
                    while (bs <= be) {
                        char ch = *bs++;
                        // check escape
                        if (ch == '\\') {
                            ch = get_escape_after_slash(*bs++);
                            if ( ch == 0) {
                                fprintf(stderr, "error found for escape of (%c) within the quoted string\n", *(bs-1));
                                return nullptr;
                            }
                            *p++ = ch;
                            continue;
                        }
                        *p++ = ch;
                        if (ch == quoteChar())  {
                            break;
                        }
                    }
                    if (bs > be) {
                        fprintf(stderr, "quote mismatch!\n");
                        return nullptr;
                    }

                    // out of a quote, bs pointing to char next to the closing "
                    continue;
                }
                /*
                if (stringQuoted() && (*bs == ',')) {
                    // change the CM to NL + CM
                    *p++ = NL;
                }
                */
                *p++ = get_spchar_map(*bs);
            }
        }
        ++bs;
        continue;
    }

    char *ret = p-1;
    while (ret >= ps) {
        if (isSpace(*ret)) {
            --ret;
            continue;
        }
        break;
    }
    return ret;
}

const char* ConfigureReader::match(const char* s, const char* e, char match_char) const {
    // matches the next 'match_char' with open/close state matched in the middle
    // return the position of the matching charactor, or nullptr if not found
    // matching '{' and '}', '[' and ']' in the middle.
    // OC/CC , OB/CB are the special charactors replaced from scan, they are replaced
    // because we want to use those charactors in the key/value, with escape'\'
    //
    // A slight complication is the escape of white space.  When escaped,
    // they need not be stripped as leading/trailing spaces, and therefore the escaped
    // white space has a special code.
    const char* p = s;
    while (p <= e) {
        if (*p == match_char) {
            return p;
        };
        if (*p == OC) {
            p = match(p+1, e, CC);
        } else if (*p == OB) {
            p = match(p+1, e, CB);
        } else if ((*p == CB) || (*p == CC)) {
            fprintf(stderr, "Closing (squre/curly) bracket unexpected at %s\n", std::string(s, p-s+1).c_str());
            return nullptr;
        }
        if (!p) {
            return nullptr;
        }
        ++p;
    }
    // failed to match
    // return e+1 if it's CM ',', for not final element, or
    // for NL, a new value could be defined without a new line, such as the last value in file
    if ((match_char == CM) || (match_char == get_spchar_map(KVDELIMITER))) {
        return e+1;
    }
    fprintf(stderr, "failed to match %c given the string: %s\n", (int)match_char, std::string(s, e-s+1).c_str());
    return nullptr;
}

const char* ConfigureReader::findValue(const char* s, const char* e, const char*&ks, const char*&ke, char end_char) const {
    // skipping initial white spaces, including special NE
    while (s<=e) {
        if (!isSpace(*s)) {
            break;
        }
        ++s;
    }
    if (s>e) {
        // empty value/key not allowed
        fprintf(stderr, "empty value not allowed, starting from %s!\n", s);
        return nullptr;
    }
    // match the end_char, strip the value, return the position after the end_char
    ks = s;
    ke = match(s, e, end_char);
    if (!ke) {
        return nullptr;
    }
    const char* ns = ke+1;
    --ke;
    strip(ks, ke);
    return ns;
}

const char* ConfigureReader::findKey(const char* s, const char* e, const char*&ks, const char*&ke) const {
    const auto* p = findValue(s, e, ks, ke, EQ);
    if (!p) return nullptr;

    // check ks and ke doesn't have NL
    const char* kp = ks;
    while (kp <= ke) {
        if (*kp == NL) {
            fprintf(stderr, "ill-formed key %s from %s\n", std::string(ks, ke-ks+1).c_str(), std::string(s, e-s+1).c_str());
            return nullptr;
        }
        ++kp;
    }
    return p;
}

bool ConfigureReader::parseValue(const char* s, const char* e, ConfigureReader::Value& val) const {
    // string between [s, e] already been stripped with white spaces. it represents a simple value or,
    // if *s starts with '{', parse composition in val as _kv
    // if *s starts with '[', recursively parse each value delimitered by ',' into val as _array
    // empty string is not allowed
    if (s>e) {
        val.clear();
        fprintf(stderr, "Got an empty value!\n");
        return false;
    }

    switch (*s) {
    case OC :  // '{'
    {
        // parse a composite value, writes to _kv
        return parseKeyValue(s+1, e-1, val._kv);
    }
    case OB :  // '['
    {
        // parse an array of values, writes to _array
        s+=1;
        auto & arr (val._array);
        while (s<=e) {
            const char *ks, *ke;

            // e-1 is the ending position without the closing ']'
            // look for ',' or ending at e-1, i.e. last element
            //
            // debug
            /*
            if (*s == '3') {
                fprintf(stderr,"here!");
            }
            */

            const char* s1 = findValue(s, e-1, ks, ke, CM);
            if (!s1) {
                // failed to match, just return nullptr
                fprintf(stderr, "failed to find a value in an array starting from %s\nparsed so far: %s\n", s, toString().c_str());
                return false;
            }
            s = s1;
            arr.emplace_back(*this);
            if (!parseValue(ks, ke, *arr.rbegin())) {
                fprintf(stderr, "failed to parse a value in a string of %s\nparsed so far: %s", std::string(ks, ke-ks+1).c_str(), toString().c_str());
                return false;
            }
        }
        return true;
    }
    default :
    {
        val.clear();
        val._value = toConfigString(s, e);
        return checkValue(val._value);
    }
    }
}

bool ConfigureReader::parseKeyValue(const char* s, const char* e, ConfigMapType& kv) const {
    // this is the root entrance from reset(ps, pe, m_value._kv)
    // iteratively gets key, and value
    const char* ks, *ke;
    while (s < e) {
        // skipping white space
        if (isSpace(*s)) {
            ++s;
            continue;
        }
        // find a key before the EQCHAR
        s = findKey(s, e, ks, ke);
        if (!s) {
            fprintf(stderr,"failed to find the key from %s\n", std::string(s, e-s+1).c_str());
            return false;
        }

        // get a key string from [ks,ke]
        std::string k = toConfigString(ks,ke);
        k = stripQuotedString(k);
        if (k.size() == 0) {
            fprintf(stderr, "empty key is found before %s\n", std::string(s, e-s+1).c_str());
            return false;
        };

        // get a string before encoded kv delimiter
        // native is NL, json is CM
        Value v(*this);
        const char *vs, *ve;
        s = findValue(s, e, vs, ve, get_spchar_map(KVDELIMITER));
        if (!s) {
            return false;
        }
        if (!parseValue(vs, ve, v)) {
            return false;
        }
        kv.insert({k,v});
    }
    return true;
}

// read the configuration file
std::string ConfigureReader::reset(const std::string& configFileName) {
    if (configFileName.size() == 0) {
        if (m_configFileName.size() == 0) {
            fprintf(stderr, "No config file name specified.  Loaded an empty configure.\n");
            // return an empty value, subsequent get will fail if its an error situation.
            // allowing it for test purpose
            return "";
            //throw std::runtime_error("Reload Failed: no config file name specified.  Reloading a temporary value?");
        };
    } else {
        m_configFileName = configFileName;
    }
    std::string ret = "";
    char *buf = nullptr;
    try {
        buf = readFile(m_configFileName.c_str());
        if (!buf) {
            return "config file read error: " + m_configFileName;
        }
        size_t bsz = strlen(buf);
        const char* pe = scan(buf, buf+bsz-1);
        if (!pe || (pe <= buf)) {
            throw std::runtime_error("Failed to scan or got empty config " + m_configFileName);
        }
        // removing encoded open/close brackets if any
        std::string cfg = stripChars(std::string(buf, pe-buf+1), OC, CC);
        const char* ps = cfg.c_str();
        pe = ps + cfg.size()-1;
        if (!parseKeyValue(ps, pe, m_value._kv)) {
            throw std::runtime_error("Failed to read config file " + m_configFileName);
        }
    } catch (const std::exception& e) {
        ret = std::string("Exception Received: ") + e.what();
    }
    free (buf);
    return ret;
}

/*
 * Output Values to config file in native or json format
 */
std::string ConfigureReader::toString(size_t start_indent) const {
    std::string ret = m_value.toString(start_indent, false);
    if (!stringQuoted()) {
        ret = stripChars(ret, '{', '}');
    }
    return ret;
}

std::string ConfigureReader::Value::toString(int indent, bool array_value) const {
    const char eqchar = _reader.EQCHAR;
    const std::string& keywrap = _reader.KEYWRAP;
    const char kv_delimiter = _reader.KVDELIMITER;

    std::string kvd_str = (kv_delimiter=='\n'? std::string("\n"): std::string(1,kv_delimiter) + "\n"); // '\n' or ',\n'
    // take care of the indent using value()
    if (_kv.size()>0) {
        // iterate composite with indentation
        std::string ret = (array_value?"":("\n" + std::string(indent, ' '))) + "{\n";
        for (const auto& kv : _kv) {
            ret += std::string(indent + 2, ' ');
            ret += (keywrap + _reader.escapeString(kv.first) + keywrap);
            ret += (std::string(" ") + std::string(1,eqchar) + std::string(" "));
            ret += kv.second.toString(indent + 4, false);
            ret += kvd_str;
        }
        if (kvd_str.size() > 1) {
            // replace the last ",\n" with "\n"
            auto sz = ret.size();
            ret = ret.substr(0, sz-kvd_str.size());
            ret += "\n";
        }
        ret += (std::string(indent, ' ') + "}");
        return ret;
    } else if (_array.size()>0) {
        std::string ret = (array_value?"":("\n" + std::string(indent, ' '))) + "[";
        if (arraySize()>0) {
            ret += ("\n" + std::string(indent+2, ' '));
            ret += _array[0].toString(indent + 2, true);
            // iterate array values
            for (size_t i = 1; i<_array.size(); ++i) {
                ret += ",\n";
                ret += std::string(indent+2, ' ');
                ret += _array[i].toString(indent + 2, true);
            }
            ret += ("\n" + std::string(indent, ' '));
        }
        ret += "]";
        return ret;
    }

    // a simple value
    if (_value.size()>0) {
        std::string ret = _value;
        if (!_reader.stringQuoted()) {
            // native case
            ret = _reader.escapeString(ret);
        } else {
            if (ret[0] == _reader.quoteChar()) {
                // string type, escape the string within the quote
                ret = keywrap + _reader.escapeString(ret.substr(1, ret.size()-2)) + keywrap;
            } else {
                // other type, no quote wrapped
                // no need to do any conversion
                // ret = escapeString(ret);
            }
        }
        return ret;
    }

    // empty value (?)
    fprintf(stderr, "empty value detected at indent %d\n", indent);
    return "";
}


/*
 * Query related functions at ConfigureReader::Value
 */
bool ConfigureReader::Value::getKeyItems(const std::string& key, std::vector<ConfigureReader::Value::KeyItem>& ki) const {
    if (key.size()==0) {
        return true;
    }
    switch (key[0]) {
    case '.' :
        return getKeyItems(key.substr(1), ki);
    case '[' :
    {
        // get the index
        size_t cbp;
        int idx;
        const auto& sstr(key.substr(1));
        try {
            idx = std::stoi(sstr, &cbp);
            // look for the close bracket
            while (cbp < sstr.size()) {
                if (sstr[cbp] == ' ') ++cbp;
                if (sstr[cbp] == ']') break;
                throw std::invalid_argument("invalid char in index brackets");
            }
            if (cbp >= sstr.size()) {
                throw std::invalid_argument("no close bracket found");
            }
        } catch (const std::invalid_argument& e) {
            fprintf(stderr, "Failed to parse query %s:  %s\n",
                    key.c_str(), e.what());
            return false;
        }
        std::string next_key = sstr.substr(cbp+1);
        ki.emplace_back("", idx);
        return getKeyItems(next_key, ki);
    }
    default :
    {
        // if . or [ found, get down the tree
        auto dotp = key.find('.');
        auto sbp = key.find('[');
        std::string k, next_key;
        if ((dotp != std::string::npos) || (sbp != std::string::npos)) {
            if ( ( dotp == std::string::npos ) ||
                 ( (sbp != std::string::npos) && (sbp < dotp) )
               )
            {
                // go with '['
                k = key.substr(0, sbp);
                next_key = key.substr(sbp);
            } else {
                // go with '.'
                k = key.substr(0, dotp);
                next_key = key.substr(dotp);
            }
        } else {
            // use the entire string as key
            k = key;
            next_key = "";
        }
        ki.emplace_back(k, 0);
        return getKeyItems(next_key, ki);
    }
    }
}

const ConfigureReader::Value*  ConfigureReader::Value::query(const std::string& key) const {
    std::vector<KeyItem> ki;
    if (!getKeyItems(key, ki)) {
        return nullptr;
    }

    const auto* v = this;
    for (const auto& it: ki) {
        const auto& k(it._key);
        const auto& i(it._idx);
        if (k.size()>0) {
            // look for key
            const auto iter = v->_kv.find(k);
            if (iter == v->_kv.end()) {
                fprintf(stderr, "key %s (in %s) not found in %s\n", k.c_str(), key.c_str(), toString(0, true).c_str());
                return nullptr;
            }
            v = &(iter->second);
        } else {
            // look for array
            if (i >=(int) v->_array.size()) {
                fprintf(stderr, "idx %d (in %s) exceeds the size of %d\n", i, key.c_str(), (int)v->_array.size());
                return nullptr;
            }
            v = &(v->_array[i]);
        }
    }
    return v;
}

bool ConfigureReader::Value::operator==(const ConfigureReader::Value& val) const {
    if (_value.size() > 0) return _value == val._value;
    if (_array.size() > 0) return _array == val._array;
    if (_kv.size() > 0)
    {
        for (const auto& kv: _kv) {
            const auto& k (kv.first);
            const auto& v (kv.second);
            const auto iter = val._kv.find(k);
            if (iter == val._kv.end()) {
                return false;
            }
            if ( ! (v == iter->second) ) {
                return false;
            }
        }
        // the other way around
        for (const auto& kv: val._kv) {
            const auto& k (kv.first);
            const auto& v (kv.second);
            const auto iter = _kv.find(k);
            if (iter == _kv.end()) {
                return false;
            }
            if ( ! (v == iter->second) ) {
                return false;
            }
        }
        return true;
    }
    return (val._value.size() == 0) &&
           (val._array.size() == 0) &&
           (val._kv.size() == 0);
}

// modifiers
ConfigureReader::Value* ConfigureReader::Value::addKey(const std::string& key, bool over_write){
    // recursively add key to this Value object
    // key could be specified in similar way with query, except the keys and indexes
    // are added.
    // if over_write is false, it throws if a key being added exists.
    //
    // return a pointer to Value object associated with the key
    // to be modified.
    // nullptr if the key cannot be parsed, i.e. '[]' not matching

    std::vector<KeyItem> ki;
    if (!getKeyItems(key, ki)) {
        return nullptr;
    }

    auto* v = this;
    for (const auto& it:ki) {
        const auto& k(it._key);
        const auto& i(it._idx);
        if (k.size()>0) {
            // adding a new key k
            auto iter = v->_kv.find(k);
            if ((iter != v->_kv.end()) && (!over_write)) {
                fprintf(stderr, "not overwrite key %s (in %s)! It exists in %s\n", k.c_str(), key.c_str(), toString(0, true).c_str());
                return nullptr;
            }
            v->_array.clear();
            v->_value.clear();
            v->_kv.insert({k,Value(_reader)});
            v = &(v->_kv.find(k)->second);
        } else {
            // adding an array value
            while (i >= (int) v->_array.size()) {
                v->addArrayElement();
            }
            v = &(v->_array[i]);
        }
    }
    return v;
}


/*
 * helper functions
 */
bool ConfigureReader::checkValue(const std::string& value) const {
    if (stringQuoted()) {
        if (value[0] == KEYWRAP[0]) {
            if (value[value.size()-1] != KEYWRAP[0]) {
                return false;
            }
            return true;
        }
        try {
            // need to be a number or true/false
            if ((value == "true") || (value == "false")) {
                return true;
            }
            size_t pos;
            std::stod(value, &pos);
            if (pos == value.size()) {
                return true;
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "Exception: %s\n", e.what());
        }
        fprintf(stderr, "invalid value encountered: %s\n", value.c_str());
        return false;
    }
    return true;
}

void ConfigureReader::set_escape_map() {
    // set m_spchar_map to be encoded special char
    memset(m_spchar_map, 0, sizeof(m_spchar_map));
    for (int i = 0; i<256; ++i) {
        m_spchar_map[i] = (char)i;
    };
    m_spchar_map[(int)EQCHAR] = EQ;
    m_spchar_map[(int)'{'] = OC;
    m_spchar_map[(int)'}'] = CC;
    m_spchar_map[(int)'['] = OB;
    m_spchar_map[(int)']'] = CB;
    m_spchar_map[(int)','] = CM;
    m_spchar_map[(int)'\n']= NL;

    if (!stringQuoted()) {
        // when scaning, input strings are allowed to 
        // read '\' + the first and will be 
        // transferred to the second
        m_read_escape.emplace( EQCHAR, EQCHAR );
        m_read_escape.emplace( '{', '{' );
        m_read_escape.emplace( '}', '}' );
        m_read_escape.emplace( '[', '[' );
        m_read_escape.emplace( ']', ']' );
        m_read_escape.emplace( ',', ',' );
        m_read_escape.emplace( '#', '#' );
        m_read_escape.emplace( '\n', '\n' );
        m_read_escape.emplace( 'n', '\n' );
        m_read_escape.emplace( ' ', WS ); // WS will be changed to ' ' in toConfigString()
        m_read_escape.emplace( '\\', '\\' );

        // when writing a string, escape the first
        // to '\' + the second
        m_write_escape.emplace( EQCHAR, EQCHAR );
        m_write_escape.emplace( '{', '{' );
        m_write_escape.emplace( '}', '}' );
        m_write_escape.emplace( '[', '[' );
        m_write_escape.emplace( ']', ']' );
        m_write_escape.emplace( ',', ',' );
        m_write_escape.emplace( '#', '#' );
        m_write_escape.emplace( '\n', 'n' );
        m_write_escape.emplace( ' ', ' ' );
        m_write_escape.emplace( '\\', '\\' );
    } else {
        // json style quoted string only allow
        // escape of "\n
        m_read_escape.emplace( '"', '"' );
        m_read_escape.emplace( '\\', '\\' );
        m_read_escape.emplace( 'n', '\n' );

        m_write_escape.emplace( '"', '"' );
        m_write_escape.emplace( '\\', '\\' );
        m_write_escape.emplace( '\n', 'n' );
    }
}

char ConfigureReader::get_spchar_map(char ch) const {
    return m_spchar_map[(int)ch];
}

std::string ConfigureReader::toConfigString (const char* ks, const char* ke) const {
    // entertain unescaped special charactors in either key or value.
    // TODO - this is only needed for unquoted native version, shouldn't happen
    // in json, since strings are quoted and should escape properly within (such as \"n)
    // that already been substituded in the scan
    std::string ret;
    while (ks<=ke) {
        char c = *ks;
        if (stringQuoted()) {
            // special chars like EQ/CM/OC shouldn't be parsed as special within quoted string
        } else {

            // non-quoted string could have such cases, like the ',' in a key, or a '=' in a value, that 
            // was parsed into special during scan, but later parsed as a part in string. For example
            // aa ,b = c==a{]d
            // this would parse into key="aa ,b", value="c==a{]d"
            // However using quote, "aa ,b" = "c++a{]d", those special chars won't be parsed as special in scan,
            // therefore no need for the conversions below.
            if (*ks == WS) {
                c = ' ';
            } else if (*ks == EQ) {
                c = EQCHAR;
            } else if (*ks == CM) {
                c = ',';
            } else if (*ks == NL) {
                c = '\n';
            } else if (*ks == OC) {
                c = '{';
            } else if (*ks == CC) {
                c = '}';
            } else if (*ks == OB) {
                c = '[';
            } else if (*ks == CB) {
                c = ']';
            } else if (((int)*ks < 32) && (*ks!='\n') &&(*ks!='\t')) {
                // we shouldn't see any unescaped special chars, other than those
                // note when outputing, these will be escaped, since they are in the escape_table
                fprintf(stderr, "special char (%d) detected in key/value of %s\n", (int)*ks, std::string(ks, ke-ks+1).c_str());
                throw std::runtime_error("special char detected in key/value " + std::string(ks, ke-ks+1));
            }
        }
        ret.push_back(c);
        ++ks;
    }
    return ret;
}

std::string ConfigureReader::stripChars(const std::string& lines, char open_char, char close_char) const {
    if (isSpace(open_char) || isSpace(close_char)) {
        throw std::runtime_error("open/close char cannot be a space!");
    }
    std::string ret;
    size_t sz =lines.size();
    // skip space
    size_t i = 0;
    for (; i<lines.size(); ++i) {
        if (isSpace(lines[i])) {
            continue;
        }
        break;
    }
    // if input all white space
    if (i >= lines.size()) {
        return "";
    }
    // if open_char is matched
    if (lines[i] != open_char) {
        return lines.substr(i, sz-i);
    }
    // find for close_char in reverse
    size_t k = sz-1;
    for (; k>i; --k) {
        if (isSpace(lines[k])) {
            continue;
        }
        break;
    }
    if (lines[k] != close_char) {
        fprintf(stderr, "stripChars() of %s: ending char %c not found for open char %c\n", lines.c_str(), close_char, open_char);
        throw std::runtime_error("end char not found for open char");
    }
    return lines.substr(i+1,k-i-1);
}

std::string ConfigureReader::stripQuotedString (const std::string& quoted_string) const {
    // remove the first and last KEYWRAP if possible, 
    // i.e. make "symbols" to symbols, removing the KEYWRAP = "\""
    // assuming key_string already stripped with spaces, new lines etc
    size_t sz = quoted_string.size();
    if ((sz == 0) || (!stringQuoted())) {
        return quoted_string;
    }
    if (quoted_string[0] == quoteChar()) {
        // check the pairing and length
        if ((sz < 3) || (quoted_string[sz-1] != quoteChar())) {
            throw std::runtime_error("Invalid key in removing wrap! KEY: "  + quoted_string + " KEYWRAP: " + KEYWRAP + "\n" + toString());
        }
        return quoted_string.substr(1,sz-2);
    }
    return quoted_string;
}

void ConfigureReader::strip(const char*& ks, const char*& ke) const {
    // skip space, tab and newlines
    // Note : leading or trailing spaces cannot be escaped
    // should we support it (?)
    while (ks <= ke) {
        if (isSpace(*ks)) 
            ++ks;
        else 
            break;
    }
    while (ks <= ke) {
        if (isSpace(*ke)) 
            --ke;
        else 
            break;
    }
}

}
