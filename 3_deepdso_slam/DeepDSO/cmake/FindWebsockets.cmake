# - Try to find LibWebsockets
# Once done this will define
#
#  WEBSOCKETS_FOUND - system has LibWebsockets
#  WEBSOCKETS_INCLUDE_DIRS - the LibWebsockets include directory
#  WEBSOCKETS_LIBRARIES - Link these to use LibWebsockets

find_package(PkgConfig)
pkg_check_modules(PC_WEBSOCKETS libwebsockets)

find_path(WEBSOCKETS_INCLUDE_DIR libwebsockets.h
          HINTS ${PC_WEBSOCKETS_INCLUDEDIR} ${PC_WEBSOCKETS_INCLUDE_DIRS}
          PATH_SUFFIXES libwebsockets)

find_library(WEBSOCKETS_LIBRARY NAMES websockets
             HINTS ${PC_WEBSOCKETS_LIBDIR} ${PC_WEBSOCKETS_LIBRARY_DIRS})

set(WEBSOCKETS_LIBRARIES ${WEBSOCKETS_LIBRARY})
set(WEBSOCKETS_INCLUDE_DIRS ${WEBSOCKETS_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Websockets DEFAULT_MSG
                                  WEBSOCKETS_LIBRARY WEBSOCKETS_INCLUDE_DIR)

mark_as_advanced(WEBSOCKETS_INCLUDE_DIR WEBSOCKETS_LIBRARY)