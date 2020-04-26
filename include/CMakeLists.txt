set(INCLUDE_DIR "/usr/include")

INSTALL(FILES
	agent.cpp
	ees_engine.cpp
	caffe_deep_network.cpp
    DESTINATION ${INCLUDE_DIR}/ADNN
    COMPONENT headers
)


