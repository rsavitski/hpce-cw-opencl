SHELL = /bin/bash

#CC = clang++
CC = g++
CFLAGS = -std=c++11 -Wall -I include -I opencl_sdk/include/
LDLIBS = -lOpenCL

%: src/rs5010/%.cpp src/heat.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%: src/%.cpp src/heat.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
