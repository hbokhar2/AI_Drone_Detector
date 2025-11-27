cmake_minimum_required(VERSION 3.24)
set(PROJECT_NAME DroneDetection)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../Build")

add_executable(${PROJECT_NAME} 
	Main.cpp
	CameraConsumer/CameraConsumer.cpp
	CameraProducer/CameraProducer.cpp
	RingBufferQueue/RingBufferQueue.cpp
	YoloProcessor/YoloProcessor.cpp
)

set(OPENCV_INCLUDE
	"${CMAKE_SOURCE_DIR}/../Libs/opencv/install/include/opencv4/"
)

if(UNIX)
	file(GLOB OPENCV_LIBS
		"${CMAKE_SOURCE_DIR}/../Libs/opencv/install/lib/*.so"
	)
elseif(WIN32)
	file(GLOB OPENCV_LIBS
		"${CMAKE_SOURCE_DIR}/../Libs/opencv/install/lib/*.lib"
	)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
	${OPENCV_INCLUDE}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
	${OPENCV_LIBS}
)
