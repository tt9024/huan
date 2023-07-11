//
// Created by Joren Gaucher on 1/24/23.
//

#pragma once

#ifndef KISCO_CEXCEPT_H
#define KISCO_CEXCEPT_H

#ifdef __cpp_noexcept_function_type
#define CNOEXCEPT noexcept
#define CEXCEPT(...) noexcept(false)
#else
#define CNOEXCEPT throw()
#define CEXCEPT(...) throw(__VA_ARGS__)
#endif

#endif //KISCO_CEXCEPT_H
