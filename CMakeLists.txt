#Read Binary Data File CMAKE
cmake_minimum_required(VERSION 2.8.12)

#Read Binary Data File CMAKE
cmake_minimum_required(VERSION 2.8.12)

set( CMAKE_SUPPRESS_REGENERATION true )
set( CMAKE_DEBUG_POSTFIX _D)  # for Debugging, put a D on the end

add_definitions( -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)

# Name the project from the current folder name
get_filename_component(CURRENTPROJECT ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(Volley)


# Other libraries
set( OPENCV_INCLUDE_DIR "$ENV{OPENCV_DIR}/include" )

#output directory for project
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )

foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} )
    set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
    set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )


# Opencv
set(LIB_CVCORE "$ENV{OPENCV_DIR}/x64/vc15/lib/opencv_core452.lib")
set(LIB_CVCORE_D "$ENV{OPENCV_DIR}/x64/vc15/lib/opencv_core452d.lib")
set(LIB_CVPROC "$ENV{OPENCV_DIR}/x64/vc15/lib/opencv_imgproc452.lib")
set(LIB_CVPROC_D "$ENV{OPENCV_DIR}/x64/vc15/lib/opencv_imgproc452d.lib")
set(LIB_CVHIGH "$ENV{OPENCV_DIR}/x64/vc15/lib/opencv_highgui452.lib")
set(LIB_CVHIGH_D "$ENV{OPENCV_DIR}/x64/vc15/lib/opencv_highgui452d.lib")
set(LIB_CVCODE "$ENV{OPENCV_DIR}/x64/vc15/lib/opencv_imgcodecs452.lib")
set(LIB_CVCODE_D "$ENV{OPENCV_DIR}/x64/vc15/lib/opencv_imgcodecs452d.lib")
set(LIB_CVVIDE "$ENV{OPENCV_DIR}/x64/vc15/lib/opencv_videoio452.lib")
set(LIB_CVVIDE_D "$ENV{OPENCV_DIR}/x64/vc15/lib/opencv_videoio452d.lib")



# Set Startup project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT Volley)


add_executable (Volley main.cpp)

include_directories( 
	
	${OPENCV_INCLUDE_DIR}
	
)
target_link_libraries(Volley
	
	optimized ${LIB_CVCORE} debug ${LIB_CVCORE_D}
	optimized ${LIB_CVPROC} debug ${LIB_CVPROC_D}
	optimized ${LIB_CVHIGH} debug ${LIB_CVHIGH_D}
	optimized ${LIB_CVCODE} debug ${LIB_CVCODE_D}
	optimized ${LIB_CVVIDE} debug ${LIB_CVVIDE_D}
	
)

