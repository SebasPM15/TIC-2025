# Versión FINAL de CMakeLists.txt para vcpkg
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(DeepDSO_Client)

# --- Definiciones para compatibilidad con Windows/MSVC ---
if(MSVC)
    add_definitions(-D_USE_MATH_DEFINES)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DBOOST_BIND_GLOBAL_PLACEHOLDERS)
endif()

# --- Configuración General ---
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Búsqueda de Dependencias (MODO CONFIG) ---
find_package(OpenCV CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS system thread)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED) # <-- NUEVO: Buscamos Sophus con vcpkg

# --- Inclusión de Directorios ---
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/util  # <-- NUEVO: Añadido para encontrar dirent.h
    # ${PROJECT_SOURCE_DIR}/thirdparty/Sophus # <-- ELIMINADO: Ya no usamos la versión antigua
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS} # <-- NUEVO: Usamos la ruta de la nueva versión
)

# --- Definición del Código Fuente ---
set(dso_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystem.cpp
    ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptimize.cpp
    ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptPoint.cpp
    ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemDebugStuff.cpp
    ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemMarginalize.cpp
    ${PROJECT_SOURCE_DIR}/src/FullSystem/Residuals.cpp
    ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseTracker.cpp
    ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseInitializer.cpp
    ${PROJECT_SOURCE_DIR}/src/FullSystem/ImmaturePoint.cpp
    ${PROJECT_SOURCE_DIR}/src/FullSystem/HessianBlocks.cpp
    ${PROJECT_SOURCE_DIR}/src/FullSystem/PixelSelector2.cpp
    ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctional.cpp
    ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedTopHessian.cpp
    ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedSCHessian.cpp
    ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctionalStructs.cpp
    ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
    ${PROJECT_SOURCE_DIR}/src/util/Undistort.cpp
    ${PROJECT_SOURCE_DIR}/src/util/globalCalib.cpp
    ${PROJECT_SOURCE_DIR}/src/util/dirent.cpp
)

# --- Creación del Ejecutable Principal ---
add_executable(dso_dataset ${PROJECT_SOURCE_DIR}/src/main_dso_pangolin.cpp ${dso_SOURCE_FILES})

# --- Enlace de Librerías ---
target_link_libraries(dso_dataset
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    Sophus::Sophus # <-- NUEVO: Enlazamos con la librería Sophus moderna
)

# Mensaje de finalización
message(STATUS "--- CMake configuration and build should be successful now! ---")