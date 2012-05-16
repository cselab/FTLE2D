This document is written by Christian Conti cconti@mavt.ethz.ch.
===============================================

REQUIREMENTS

to use FTLE2D, VTK is required
optionally, to run on GPU, OpenCL is required

===============================================

MAKE

to generate the program, the file make.platform should be modified with the necessary paths to VTK libraries and headers.
Additionally, compilation flags and other options can be set in the same file.

make FTLE CC=gcc
	compiles the stand-alone code with gcc

make libFTLE opencl=0 CC=gcc
	compiles the library version with gcc, without opencl

===============================================

RUN

"./FTLE -help" displays on the terminal the runtime options:
usage:
	-help				display this message
	-verbose [0|1]			set to verbose (default: false)
	-sx N				set problem size in x direction
	-sy N				set problem size in y direction
	-CellCentered [0|1]		data is cell centered (default: false)
	-nFields N			set number of velocity fields
	-nFTLEs N			set number of desired FTLE fields-1
	-OpenCL [0|1]			run OpenCL implementation (default: false)
	-device N			run on device rank N (default: 0)
	--ftle-kernel-folder path	set path to OpenCL kernel files (default: ../../FTLE/include/)
	-type [brunton|overlap]		select method (default: fallback)
	-onhost [0|1]			yse CL_MEM_ALLOC_HOST_PTR for image allocation, only for brunton (default: false)
	-profilecl [0|1]		print OpenCL profiling information (default: false)
	-GFLOPs [0|1]			print GFLOPs information (default: false)

===============================================

PROBLEM SETTINGS

the velocity fields are generated from the runProblem method that can be found in ProblemSettings.h

===============================================