SHELL := /bin/bash

UNAME?=$(shell uname)
HOSTNAME?=$(shell hostname)
config ?= release
vtk ?= 0

#===========================
#	Includes
#===========================

include make.includes

#===========================
#	Compiler
#===========================

CC ?= gcc
LD = $(CC)

#===========================
#	Source Files
#===========================

SOURCES += $(notdir ../source/FTLE2D.cpp)
SOURCES += $(notdir ../source/Profiler.cpp)
SOURCES += $(notdir ../source/M2P_CPU_SSE.cpp)
SOURCES += $(notdir ../source/main.cpp)

ifeq "$(config)" "release"
	CFLAGS += -DNDEBUG
endif

ifeq "$(opencl)" "1"
	SOURCES += $(notdir ../source/FTLE_OpenCL.cpp)
	SOURCES += $(notdir ../source/FTLE_OpenCL_OverlapGPU.cpp)
	SOURCES += $(notdir ../source/FTLE_OpenCL_Brunton.cpp)
	SOURCES += $(notdir ../source/EngineCL.cpp)
	CFLAGS += -D_OPENCL
endif

ifeq "$(MAKECMDGOALS)" "libFTLE"
	CFLAGS += -fPIC -D_LIBRARY

	ifeq "$(UNAME)" "Darwin"
		SO = -dynamiclib -install_name libFTLE.1.dylib -o libFTLE.1.0.dylib
		LN1 = ln -s libFTLE.1.0.dylib libFTLE.1.dylib
		LN2 = ln -s libFTLE.1.0.dylib libFTLE.dylib
		CP = cp libFTLE.*dylib ../lib/
	else
		SO = -shared -Wl,-soname,libFTLE.so.1 -o libFTLE.so.1.0
		LN1 = ln -s libFTLE.so.1.0 libFTLE.so.1
		LN2 = ln -s libFTLE.so.1 libFTLE.so
		CP = cp libFTLE.so* ../lib/
	endif
	
endif

ifeq "$(opencl)" "1"

	ifeq "$(UNAME)" "Darwin"
		LIBRARIES += -framework OpenCL
	else
		LIBRARIES += -lOpenCL
	endif
	
	CFLAGS += -D __CL_ENABLE_EXCEPTIONS
endif

ifneq "$(MAKECMDGOALS)" "libFTLE"
ifeq "$(vtk)" "1"
	CFLAGS += -D_WITH_VTK_
	LIBRARIES += -lvtkVolumeRendering \
		-lvtkRendering \
		-lvtkIO \
		-lvtkGenericFiltering \
		-lvtkGraphics \
		-lvtkImaging \
		-lvtkFiltering \
		-lvtkCommon \
		-lvtkftgl \
		-lvtkDICOMParser \
		-lvtksys \
		-lvtkexoIIc \
		-lvtkNetCDF \
		-lvtkNetCDF_cxx \
		-lvtkmetaio \
		-lvtksqlite \
		-lvtkverdict \
		-lvtkexpat \
		-lvtktiff \
		-lvtkpng \
		-lvtkjpeg \
		-lvtkzlib \
		-lvtkHybrid \
		-lvtkfreetype
endif
endif

LIBRARIES += -lstdc++

INCLUDES += -I../source/

#===========================
#	Build Rules
#===========================

BUILDDIR = ./
SRC_DIR = $(BUILDDIR)/../source/
OBJ_DIR = $(BUILDDIR)

OBJECTS := $(SOURCES:%.cpp=$(OBJ_DIR)/%.o)

VPATH := $(SRC_DIR):

.DEFAULT: FTLE

FTLE: $(OBJECTS)
	$(LD) $(OBJECTS) -o $@ $(LIBRARIES) $(LDFLAGS)

libFTLE: $(OBJECTS)
	mkdir -p ../include
	cp ../source/FTLE.h ../include/
	cp ../source/dumpToVTK.h ../include/
	cp ../source/*.cl ../include/
	mkdir -p ../lib
	$(LD) $(OBJECTS) $(SO) $(LIBRARIES) $(LDFLAGS) 
	$(LN1)
	$(LN2)
	$(CP)
	
%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $^ -o $@

clean:
	rm -f *.o FTLE libFTLE.so* libFTLE*dylib
