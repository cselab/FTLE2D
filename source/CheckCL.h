/*
 *  CheckCL.h
 *  FTLE2D
 *
 *  Created by Diego Rossinelli on 11/17/09.
 *  Copyright 2009 ETH Zurich. All rights reserved.
 *
 */

#define launchError(message ,  file,  line){\
printf("[%s]:%d >>%s\n", file, line, message);\
}

#define _CheckCL(expr, b) {\
cl_int res321123321 = expr;\
switch (res321123321) {\
case CL_DEVICE_NOT_FOUND: launchError("CL_DEVICE_NOT_FOUND", __func__, __LINE__); break;\
case CL_DEVICE_NOT_AVAILABLE: launchError("CL_DEVICE_NOT_AVAILABLE", __func__, __LINE__); break;\
case CL_COMPILER_NOT_AVAILABLE: launchError("CL_COMPILER_NOT_AVAILABLE", __func__, __LINE__); break;\
case CL_MEM_OBJECT_ALLOCATION_FAILURE: launchError("CL_MEM_OBJECT_ALLOCATION_FAILURE", __func__, __LINE__); break;\
case CL_OUT_OF_RESOURCES: launchError("CL_OUT_OF_RESOURCES", __func__, __LINE__); break;\
case CL_OUT_OF_HOST_MEMORY: launchError("CL_OUT_OF_HOST_MEMORY", __func__, __LINE__); break;\
case CL_PROFILING_INFO_NOT_AVAILABLE: launchError("CL_PROFILING_INFO_NOT_AVAILABLE", __func__, __LINE__); break;\
case CL_MEM_COPY_OVERLAP: launchError("CL_MEM_COPY_OVERLAP", __func__, __LINE__); break;\
case CL_IMAGE_FORMAT_MISMATCH: launchError("CL_IMAGE_FORMAT_MISMATCH", __func__, __LINE__); break;\
case CL_IMAGE_FORMAT_NOT_SUPPORTED: launchError("CL_IMAGE_FORMAT_NOT_SUPPORTED", __func__, __LINE__); break;\
case CL_BUILD_PROGRAM_FAILURE: launchError("CL_BUILD_PROGRAM_FAILURE", __func__, __LINE__); break;\
case CL_MAP_FAILURE: launchError("CL_MAP_FAILURE", __func__, __LINE__); break;\
case CL_INVALID_VALUE: launchError("CL_INVALID_VALUE", __func__, __LINE__); break;\
case CL_INVALID_DEVICE_TYPE: launchError("CL_INVALID_DEVICE_TYPE", __func__, __LINE__); break;\
case CL_INVALID_PLATFORM: launchError("CL_INVALID_PLATFORM", __func__, __LINE__); break;\
case CL_INVALID_DEVICE: launchError("CL_INVALID_DEVICE", __func__, __LINE__); break;\
case CL_INVALID_CONTEXT: launchError("CL_INVALID_CONTEXT", __func__, __LINE__); break;\
case CL_INVALID_QUEUE_PROPERTIES: launchError("CL_INVALID_QUEUE_PROPERTIES", __func__, __LINE__); break;\
case CL_INVALID_COMMAND_QUEUE: launchError("CL_INVALID_COMMAND_QUEUE", __func__, __LINE__); break;\
case CL_INVALID_HOST_PTR: launchError("CL_INVALID_HOST_PTR", __func__, __LINE__); break;\
case CL_INVALID_MEM_OBJECT: launchError("CL_INVALID_MEM_OBJECT", __func__, __LINE__); break;\
case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: launchError("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", __func__, __LINE__); break;\
case CL_INVALID_IMAGE_SIZE: launchError("CL_INVALID_IMAGE_SIZE", __func__, __LINE__); break;\
case CL_INVALID_SAMPLER: launchError("CL_INVALID_SAMPLER", __func__, __LINE__); break;\
case CL_INVALID_BINARY: launchError("CL_INVALID_BINARY", __func__, __LINE__); break;\
case CL_INVALID_BUILD_OPTIONS: launchError("CL_INVALID_BUILD_OPTIONS", __func__, __LINE__); break;\
case CL_INVALID_PROGRAM: launchError("CL_INVALID_PROGRAM", __func__, __LINE__); break;\
case CL_INVALID_PROGRAM_EXECUTABLE: launchError("CL_INVALID_PROGRAM_EXECUTABLE", __func__, __LINE__); break;\
case CL_INVALID_KERNEL_NAME: launchError("CL_INVALID_KERNEL_NAME", __func__, __LINE__); break;\
case CL_INVALID_KERNEL_DEFINITION: launchError("CL_INVALID_KERNEL_DEFINITION", __func__, __LINE__); break;\
case CL_INVALID_KERNEL: launchError("CL_INVALID_KERNEL", __func__, __LINE__); break;\
case CL_INVALID_ARG_INDEX: launchError("CL_INVALID_ARG_INDEX", __func__, __LINE__); break;\
case CL_INVALID_ARG_VALUE: launchError("CL_INVALID_ARG_VALUE", __func__, __LINE__); break;\
case CL_INVALID_ARG_SIZE: launchError("CL_INVALID_ARG_SIZE", __func__, __LINE__); break;\
case CL_INVALID_KERNEL_ARGS: launchError("CL_INVALID_KERNEL_ARGS", __func__, __LINE__); break;\
case CL_INVALID_WORK_DIMENSION: launchError("CL_INVALID_WORK_DIMENSION", __func__, __LINE__); break;\
case CL_INVALID_WORK_GROUP_SIZE: launchError("CL_INVALID_WORK_GROUP_SIZE", __func__, __LINE__); break;\
case CL_INVALID_WORK_ITEM_SIZE: launchError("CL_INVALID_WORK_ITEM_SIZE", __func__, __LINE__); break;\
case CL_INVALID_GLOBAL_OFFSET: launchError("CL_INVALID_GLOBAL_OFFSET", __func__, __LINE__); break;\
case CL_INVALID_EVENT_WAIT_LIST: launchError("CL_INVALID_EVENT_WAIT_LIST", __func__, __LINE__); break;\
case CL_INVALID_EVENT: launchError("CL_INVALID_EVENT", __func__, __LINE__); break;\
case CL_INVALID_OPERATION: launchError("CL_INVALID_OPERATION", __func__, __LINE__); break;\
case CL_INVALID_GL_OBJECT: launchError("CL_INVALID_GL_OBJECT", __func__, __LINE__); break;\
case CL_INVALID_BUFFER_SIZE: launchError("CL_INVALID_BUFFER_SIZE", __func__, __LINE__); break;\
case CL_INVALID_MIP_LEVEL: launchError("CL_INVALID_MIP_LEVEL", __func__, __LINE__); break;\
}\
if (res321123321 != CL_SUCCESS && (b)) abort();\
}

#ifndef CheckCL
#define CheckCL(expr) {_CheckCL(expr, true);}
#endif

#ifndef CheckRelaxCL
#define CheckRelaxCL(expr) {_CheckCL(expr, false);}
#endif
