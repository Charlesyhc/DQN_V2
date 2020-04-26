SET(example_sources
	freq_learning.cpp
	)
#for each source: build an executable and install
FOREACH(example_source ${example_sources})
    GET_FILENAME_COMPONENT(example_name ${example_source} NAME_WE)
    ADD_EXECUTABLE(${example_name} ${example_source})
    TARGET_LINK_LIBRARIES(${example_name} adnn)
ENDFOREACH(example_source)

