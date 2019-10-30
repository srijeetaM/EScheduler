#ifndef __STRING_H  // Control inclusion of header files
#define __STRING_H


#include <cstdio>
#include <cstring>
#include <cstdlib>

char **strsplit(const char *strn, const char *delim);
void free_list(char **list);
int list_len(char **list);

#include "string.inl"

#endif