// dirent.cpp para MSVC/Windows
#if defined(_WIN32)

#include "dirent.h"
#include <stddef.h>
#include <string.h>

DIR* opendir(const char* name) {
    if (name == NULL || name[0] == '\0') {
        return NULL;
    }

    std::string path = name;
    if (path.back() != '/' && path.back() != '\\') {
        path += "\\*";
    } else {
        path += "*";
    }

    DIR* dir = new DIR;
    dir->hFind = FindFirstFileA(path.c_str(), &(dir->findData));
    dir->first_read = true;

    if (dir->hFind == INVALID_HANDLE_VALUE) {
        delete dir;
        return NULL;
    }
    return dir;
}

struct dirent* readdir(DIR* dir) {
    if (dir == NULL || dir->hFind == INVALID_HANDLE_VALUE) {
        return NULL;
    }

    if (dir->first_read) {
        dir->first_read = false;
    } else {
        if (!FindNextFileA(dir->hFind, &(dir->findData))) {
            return NULL;
        }
    }
    
    strncpy_s(dir->current.d_name, MAX_PATH, dir->findData.cFileName, _TRUNCATE);
    return &(dir->current);
}

int closedir(DIR* dir) {
    if (dir == NULL) {
        return -1;
    }
    FindClose(dir->hFind);
    delete dir;
    return 0;
}

#endif // _WIN32