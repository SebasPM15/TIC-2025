// dirent.h para MSVC/Windows
// De https://github.com/tronkko/dirent/blob/master/include/dirent.h
#ifndef DIRENT_H
#define DIRENT_H

#if defined(_WIN32)

#include <windows.h>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dirent {
    char d_name[MAX_PATH + 1];
} dirent;

typedef struct DIR {
    HANDLE hFind;
    WIN32_FIND_DATAA findData;
    dirent current;
    bool first_read;
} DIR;

DIR* opendir(const char* name);
struct dirent* readdir(DIR* dir);
int closedir(DIR* dir);

#ifdef __cplusplus
}
#endif

#else // Si no es _WIN32, usa el dirent.h del sistema
#include <dirent.h>
#endif // _WIN32

#endif // DIRENT_H