# SHARED 
add_library(adnn SHARED
	agent.cpp
	ees_engine.cpp
	caffe_deep_network.cpp
   )

target_link_libraries(adnn ${ADNN_LINK_LIBS})

INSTALL(TARGETS adnn
    LIBRARY DESTINATION /usr/lib COMPONENT libraries # .so file
)
