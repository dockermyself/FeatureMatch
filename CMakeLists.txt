cmake_minimum_required(VERSION 3.12)

project(camera)

# Set C++ standard 
set(CMAKE_CXX_STANDARD 14)

#Debug or Release
set(CMAKE_BUILD_TYPE Debug)

# Find OpenCV package
find_package(OpenCV REQUIRED)

#Eigen
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} 
                    ${EIGEN3_INCLUDE_DIR}
                    ${CMAKE_SOURCE_DIR}/include)

# Add your source files here
set(SOURCES main.cpp OrbExtractor.cpp Frame.cpp ImuProcess.cpp ImuReader.cpp)

# Create the executable
add_executable(camera ${SOURCES})

# Link against OpenCV
target_link_libraries(camera ${OpenCV_LIBS} pthread atomic)