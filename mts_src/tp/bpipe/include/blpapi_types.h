/* Copyright 2012. Bloomberg Finance L.P.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:  The above
 * copyright notice and this permission notice shall be included in all copies
 * or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
// blpapi_types.h                                                     -*-C++-*-
#ifndef INCLUDED_BLPAPI_TYPES
#define INCLUDED_BLPAPI_TYPES

//@PURPOSE: Provide BLPAPI types.
//
//@CLASSES:
// blpapi::DataType: Data types in the schema
// blpapi::Bytes: Represents a byte array
//
//@DESCRIPTION: This component implements BLPAPI types.

typedef int blpapi_Bool_t;
typedef char blpapi_Char_t;
typedef unsigned char blpapi_UChar_t;
typedef short blpapi_Int16_t;
typedef unsigned short blpapi_UInt16_t;
typedef int blpapi_Int32_t;
typedef unsigned int blpapi_UInt32_t;
typedef long long blpapi_Int64_t;
typedef unsigned long long blpapi_UInt64_t;
typedef float blpapi_Float32_t;
typedef double blpapi_Float64_t;

enum blpapi_DataType_t {
    BLPAPI_DATATYPE_BOOL = 1 // Bool
    ,
    BLPAPI_DATATYPE_CHAR = 2 // Char
    ,
    BLPAPI_DATATYPE_BYTE = 3 // Unsigned 8 bit value
    ,
    BLPAPI_DATATYPE_INT32 = 4 // 32 bit Integer
    ,
    BLPAPI_DATATYPE_INT64 = 5 // 64 bit Integer
    ,
    BLPAPI_DATATYPE_FLOAT32 = 6 // 32 bit Floating point - IEEE
    ,
    BLPAPI_DATATYPE_FLOAT64 = 7 // 64 bit Floating point - IEEE
    ,
    BLPAPI_DATATYPE_STRING = 8 // ASCIIZ string
    ,
    BLPAPI_DATATYPE_BYTEARRAY = 9 // Opaque binary data
    ,
    BLPAPI_DATATYPE_DATE = 10 // Date
    ,
    BLPAPI_DATATYPE_TIME = 11 // Timestamp
    ,
    BLPAPI_DATATYPE_DECIMAL = 12 //
    ,
    BLPAPI_DATATYPE_DATETIME = 13 // Date and time
    ,
    BLPAPI_DATATYPE_ENUMERATION = 14 // An opaque enumeration
    ,
    BLPAPI_DATATYPE_SEQUENCE = 15 // Sequence type
    ,
    BLPAPI_DATATYPE_CHOICE = 16 // Choice type
    ,
    BLPAPI_DATATYPE_CORRELATION_ID = 17 // Used for some internal
                                        // messages
};

typedef enum blpapi_Logging_Severity_t {
    blpapi_Logging_SEVERITY_OFF = 0,
    blpapi_Logging_SEVERITY_FATAL = 1,
    blpapi_Logging_SEVERITY_ERROR = 2,
    blpapi_Logging_SEVERITY_WARN = 3,
    blpapi_Logging_SEVERITY_INFO = 4,
    blpapi_Logging_SEVERITY_DEBUG = 5,
    blpapi_Logging_SEVERITY_TRACE = 6
} blpapi_Logging_Severity_t;

struct blpapi_AbstractSession;
typedef struct blpapi_AbstractSession blpapi_AbstractSession_t;

struct blpapi_Constant;
typedef struct blpapi_Constant blpapi_Constant_t;

struct blpapi_ConstantList;
typedef struct blpapi_ConstantList blpapi_ConstantList_t;

struct blpapi_Element;
typedef struct blpapi_Element blpapi_Element_t;

struct blpapi_Event;
typedef struct blpapi_Event blpapi_Event_t;

struct blpapi_EventDispatcher;
typedef struct blpapi_EventDispatcher blpapi_EventDispatcher_t;

struct blpapi_EventFormatter;
typedef struct blpapi_EventFormatter blpapi_EventFormatter_t;

struct blpapi_EventQueue;
typedef struct blpapi_EventQueue blpapi_EventQueue_t;

struct blpapi_MessageIterator;
typedef struct blpapi_MessageIterator blpapi_MessageIterator_t;

struct blpapi_Name;
typedef struct blpapi_Name blpapi_Name_t;

struct blpapi_Operation;
typedef struct blpapi_Operation blpapi_Operation_t;

struct blpapi_ProviderSession;
typedef struct blpapi_ProviderSession blpapi_ProviderSession_t;

struct blpapi_RequestTemplate;
typedef struct blpapi_RequestTemplate blpapi_RequestTemplate_t;

struct blpapi_Service;
typedef struct blpapi_Service blpapi_Service_t;

struct blpapi_Session;
typedef struct blpapi_Session blpapi_Session_t;

struct blpapi_SessionOptions;
typedef struct blpapi_SessionOptions blpapi_SessionOptions_t;

struct blpapi_TlsOptions;
typedef struct blpapi_TlsOptions blpapi_TlsOptions_t;

struct blpapi_SubscriptionItrerator;
typedef struct blpapi_SubscriptionItrerator blpapi_SubscriptionIterator_t;

struct blpapi_Identity;
typedef struct blpapi_Identity blpapi_UserHandle;
typedef struct blpapi_Identity blpapi_UserHandle_t;

struct blpapi_Identity;
typedef struct blpapi_Identity blpapi_Identity_t;

struct blpapi_SchemaElementDefinition;
typedef struct blpapi_SchemaElementDefinition blpapi_SchemaElementDefinition_t;

struct blpapi_Datetime_tag;
typedef struct blpapi_Datetime_tag blpapi_Datetime_t;

struct blpapi_MessageFormatter;
typedef struct blpapi_MessageFormatter blpapi_MessageFormatter_t;

struct blpapi_MessageProperties;
typedef struct blpapi_MessageProperties blpapi_MessageProperties_t;

struct blpapi_AuthOptions;
typedef struct blpapi_AuthOptions blpapi_AuthOptions_t;

struct blpapi_AuthUser;
typedef struct blpapi_AuthUser blpapi_AuthUser_t;

struct blpapi_AuthApplication;
typedef struct blpapi_AuthApplication blpapi_AuthApplication_t;

struct blpapi_AuthToken;
typedef struct blpapi_AuthToken blpapi_AuthToken_t;

#ifdef __cplusplus

#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace BloombergLP {
namespace blpapi {
// ===============
// struct DataType
// ===============

struct DataType {
    // Contains the possible data types which can be represented in an
    // Element.

  public:
    enum Value {
        // The possible data types which can be represented in an Element.
        BOOL = BLPAPI_DATATYPE_BOOL, // Boolean
        CHAR = BLPAPI_DATATYPE_CHAR, // Char
        BYTE = BLPAPI_DATATYPE_BYTE, // Unsigned 8 bit value
        INT32 = BLPAPI_DATATYPE_INT32, // 32 bit Integer
        INT64 = BLPAPI_DATATYPE_INT64, // 64 bit Integer
        FLOAT32 = BLPAPI_DATATYPE_FLOAT32, // 32 bit Floating point
        FLOAT64 = BLPAPI_DATATYPE_FLOAT64, // 64 bit Floating point
        STRING = BLPAPI_DATATYPE_STRING, // ASCIIZ string
        BYTEARRAY = BLPAPI_DATATYPE_BYTEARRAY, // Opaque binary data
        DATE = BLPAPI_DATATYPE_DATE, // Date
        TIME = BLPAPI_DATATYPE_TIME, // Timestamp
        DECIMAL = BLPAPI_DATATYPE_DECIMAL, // Currently Unsupported
        DATETIME = BLPAPI_DATATYPE_DATETIME, // Date and time
        ENUMERATION = BLPAPI_DATATYPE_ENUMERATION, // An opaque enumeration
        SEQUENCE = BLPAPI_DATATYPE_SEQUENCE, // Sequence type
        CHOICE = BLPAPI_DATATYPE_CHOICE, // Choice type
        CORRELATION_ID = BLPAPI_DATATYPE_CORRELATION_ID // Used for some
                                                        // internal messages
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <typename T, typename U = bool> struct IsString : std::false_type {
};

template <typename T>
struct IsString<T,
        typename std::enable_if<
                std::is_member_function_pointer<decltype(&T::c_str)>::value,
                bool>::type> : std::true_type {
};

template <typename T, typename U = bool>
struct IsByteSequence : std::false_type {
};

template <typename T>
struct IsByteSequence<T,
        typename std::enable_if<
                std::is_member_function_pointer<
                        decltype(static_cast<const typename T::value_type *(
                                        T::*)() const>(&T::data))>::value
                        && sizeof(typename T::value_type) == 1
                        && !IsString<T>::value,
                bool>::type> : std::true_type {
};

template <typename T, typename ITER, typename U = bool>
struct IsByteContainer : std::false_type {
};

template <typename T, typename ITER>
struct IsByteContainer<T,
        ITER,
        typename std::enable_if<std::is_constructible<T, ITER, ITER>::value
                        && sizeof(typename T::value_type) == 1,
                bool>::type> : std::true_type {
};

#endif // ifndef DOXYGEN_SHOULD_SKIP_THIS

// ===========
// class Bytes
// ===========

class Bytes {
    // Bytes is used as a byte array representation.
    //
    // It contains the pointer and size of a contiguous block of memory - byte
    // array. Data is not copied into Bytes, i.e. it needs to live longer than
    // the corresponding Bytes object.

  public:
    typedef const char element_type;
    typedef char value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef const char *pointer;
    typedef const char *const_pointer;
    typedef const char& reference;
    typedef const char& const_reference;
    typedef const char *iterator;
    typedef const char *const_iterator;
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  private:
    const value_type *d_ptr; // pointer to the byte array
    size_type d_length; // size of the byte array

  public:
    Bytes() noexcept;
    // Create an empty Bytes object.

    Bytes(pointer ptr, size_type length) noexcept;
    // Create a Bytes object pointing to a contiguous block of memory starting
    // from the specified 'ptr' and having the specified 'length'.

    void assign(pointer ptr, size_type length) noexcept;
    // Replace the content of the Bytes object to make it pointing to a
    // contiguous block of memory starting from the specified 'ptr' and having
    // the specified 'length'.

    iterator begin() const noexcept;
    // Return an iterator to the first element of the byte array. If the byte
    // array is empty, the returned iterator will be equal to 'end()'.

    iterator end() const noexcept;
    // Return an iterator to the element following the last element of the byte
    // array. This element acts as a placeholder; attempting to access it
    // results in undefined behavior.

    reverse_iterator rbegin() const noexcept;
    // Return a reverse iterator to the first element of the reversed byte
    // array. It corresponds to the last element of the non-reversed byte
    // array. If the byte array is empty, the returned iterator is equal to
    // 'rend()'.

    reverse_iterator rend() const noexcept;
    // Return a reverse iterator to the element following the last element of
    // the reversed byte array. It corresponds to the element preceding the
    // first element of the non-reversed byte array. This element acts as a
    // placeholder, attempting to access it results in undefined behavior.

    reference front() const;
    // Return a reference to the first element in the byte array. Calling
    // 'front' on an empty byte array results in undefined behavior.

    reference back() const;
    // Return a reference to the last element in the byte array. Calling 'back'
    // on an empty byte array results in undefined behavior.

    reference operator[](size_type index) const;
    // Return a reference to the specified 'index'-th element of the byte
    // array. The behavior is undefined if 'index' is out of range (i.e., if it
    // is greater than or equal to 'size()').

    pointer data() const noexcept;
    // Return a pointer to the beginning of the byte array.

    size_type size() const noexcept;
    // Return the number of elements in the byte array.

    bool empty() const noexcept;
    // Check if the byte array is empty.

    Bytes first(size_type count) const;
    // Obtain a byte array that is a view over the first specified 'count'
    // elements of this byte array. The behavior is undefined if
    // 'count > size()'.

    Bytes last(size_type count) const;
    // Obtain a byte array that is a view over the last specified 'count'
    // elements of this byte array. The behavior is undefined if
    // 'count > size()'.

    Bytes subspan(size_type offset,
            size_type count = static_cast<std::size_t>(-1)) const;
    // Obtain a byte array that is a view over the specified 'count' elements
    // of this byte array starting at the specified 'offset'. If 'count == -1',
    // the number of elements in the subspan is 'size() - offset' (i.e., it
    // ends at the end of '*this'). The behavior is undefined if either
    // 'offset' or 'count' is out of range. This happens if 'offset > size()',
    // or 'count != -1' and 'count > size() - offset'.

    template <typename BYTE_CONTAINER,
            typename std::enable_if<
                    IsByteContainer<BYTE_CONTAINER, iterator>::value,
                    bool>::type
            = true>
    operator BYTE_CONTAINER() const;
    // Convert the byte array to a 'BYTE_CONTAINER'. 'BYTE_CONTAINER' is any
    // type that can be constructed with two iterators to the beginning and the
    // end of the byte array, i.e. 'BYTE_CONTAINER(begin(), end())'. Also,
    // 'BYTE_CONTAINER' must provide a type 'value_type' such that
    // 'sizeof(BYTE_CONTAINER::value_type) == 1'.
};

typedef blpapi_Bool_t Bool;
typedef blpapi_Char_t Char;
typedef blpapi_UChar_t UChar;
typedef blpapi_Int16_t Int16;
typedef blpapi_UInt16_t UInt16;
typedef blpapi_Int32_t Int32;
typedef blpapi_UInt32_t UInt32;
typedef blpapi_Int64_t Int64;
typedef blpapi_UInt64_t UInt64;
typedef blpapi_Float32_t Float32;
typedef blpapi_Float64_t Float64;

//=============================================================================
//                           INLINE FUNCTION DEFINITIONS
//=============================================================================

// -----------
// class Bytes
// -----------

inline Bytes::Bytes() noexcept
    : d_ptr(nullptr)
    , d_length(0)
{
}

inline Bytes::Bytes(pointer ptr, size_type length) noexcept
    : d_ptr(ptr)
    , d_length(length)
{
}

inline void Bytes::assign(pointer ptr, size_type length) noexcept
{
    d_ptr = ptr;
    d_length = length;
}

inline Bytes::iterator Bytes::begin() const noexcept { return d_ptr; }

inline Bytes::iterator Bytes::end() const noexcept { return d_ptr + d_length; }

inline Bytes::reverse_iterator Bytes::rbegin() const noexcept
{
    return reverse_iterator(end());
}

inline Bytes::reverse_iterator Bytes::rend() const noexcept
{
    return reverse_iterator(begin());
}

inline Bytes::reference Bytes::front() const
{
    assert(d_ptr);
    assert(d_length);
    return d_ptr[0];
}

inline Bytes::reference Bytes::back() const
{
    assert(d_ptr);
    assert(d_length);
    return d_ptr[d_length - 1];
}

inline Bytes::reference Bytes::operator[](size_type index) const
{
    assert(d_ptr);
    assert(index < d_length);
    return d_ptr[index];
}

inline Bytes::pointer Bytes::data() const noexcept { return d_ptr; }

inline Bytes::size_type Bytes::size() const noexcept { return d_length; }

inline bool Bytes::empty() const noexcept { return 0 == d_length; }

inline Bytes Bytes::first(size_type count) const
{
    assert(count <= d_length);
    return Bytes(d_ptr, count);
}

inline Bytes Bytes::last(size_type count) const
{
    assert(count <= d_length);
    return Bytes(d_ptr + d_length - count, count);
}

inline Bytes Bytes::subspan(size_type offset, size_type count) const
{
    assert(offset <= d_length);
    assert(count == static_cast<std::size_t>(-1)
            || count <= d_length - offset);
    return Bytes(d_ptr + offset,
            count == static_cast<std::size_t>(-1) ? d_length - offset : count);
}

template <typename BYTE_CONTAINER,
        typename std::enable_if<
                IsByteContainer<BYTE_CONTAINER, Bytes::iterator>::value,
                bool>::type>
inline Bytes::operator BYTE_CONTAINER() const
{
    assert(d_ptr || d_length == 0);
    return BYTE_CONTAINER(begin(), end());
}

} // close namespace blpapi
} // close namespace BloombergLP

#endif // ifdef __cplusplus

#endif // #ifndef INCLUDED_BLPAPI_TYPES
