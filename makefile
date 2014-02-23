SHELL = /bin/bash

#CC = clang++
CC = g++
CFLAGS = -std=c++11 -Wall -I include -I opencl_sdk/include/
LDLIBS = -lOpenCL

#
#step_world_v1_lambda: src/rs5010/step_world_v1_lambda.cpp src/heat.cpp
#	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
#
#step_world_v2_function: src/rs5010/step_world_v2_function.cpp src/heat.cpp
#	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
#
#step_world_v3_opencl: src/rs5010/step_world_v3_opencl.cpp src/heat.cpp
#	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
#
#step_world_v4_double_buffered: src/rs5010/step_world_v4_double_buffered.cpp src/heat.cpp
#	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
#
#make_world: src/make_world.cpp src/heat.cpp
#	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
#
#step_world: src/step_world.cpp src/heat.cpp
#	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
#
#render_world: src/render_world.cpp src/heat.cpp
#	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
#

%: src/rs5010/%.cpp src/heat.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

%: src/%.cpp src/heat.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
