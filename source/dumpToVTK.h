/*
 *  LayerToVTK.h
 *  FTLE2D
 *
 *  Created by Diego Rossinelli on 7/21/09.
 *  Copyright 2009 CSE Lab, ETH Zurich. All rights reserved.
 *
 */

#pragma once

#ifndef _DOUBLE
typedef float Real;
#else
typedef double Real;
#endif

#ifdef _WITH_VTK_
#include <vtkPoints.h> 
#include <vtkCell.h>
#include <vtkImageData.h>
#include <vtkImageNoiseSource.h>
#include <vtkFloatArray.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#else
#warning VTK SUPPORT IS DISABLED AT COMPILE TIME. USE make vtk=1 TO ACTIVATE.
#endif

inline void dump(string sFileNamePattern, int step, const Real * field, int sizeX, int sizeY, bool bBoundaries=true)
{
#ifdef _WITH_VTK_
	char buf[500];
	
	sprintf(buf, sFileNamePattern.data(), step);
	string sFileName(buf);
	
	vtkImageData * grid = vtkImageData::New();
	
	int sx = bBoundaries ? sizeX : sizeX-2;
	int sy = bBoundaries ? sizeY : sizeY-2;
	
	grid->SetExtent(0,sx-1,0,sy-1,0,0);
	grid->SetDimensions(sx, sy, 1);
	grid->SetScalarTypeToFloat();
	grid->SetNumberOfScalarComponents(1);
	grid->AllocateScalars();
	grid->SetSpacing(1./sizeX, 1./sizeX, 1);
	grid->SetOrigin(0,0,0);
	
	int offsetX = bBoundaries ? 0 : 1;
	int offsetY = bBoundaries ? 0 : 1;
	
	for(int iy=0; iy<sy; iy++)
		for(int ix=0; ix<sx; ix++)
		{
			int index = ix+offsetX + sizeX*(iy+offsetY);
			grid->SetScalarComponentFromFloat(ix, iy, 0, 0, field[index]);
		}
	
	vtkXMLImageDataWriter * writer = vtkXMLImageDataWriter::New();
	writer->SetFileName(sFileName.c_str());
	writer->SetInput(grid);
	writer->Write();
	
	writer->Delete();
	grid->Delete();
#endif
}

inline void dump(string sFileNamePattern, int step, const Real * field1, const Real * field2, int sizeX, int sizeY, bool bBoundaries=true)
{
#ifdef _WITH_VTK_
	char buf[500];
	
	sprintf(buf, sFileNamePattern.data(), step);
	string sFileName(buf);
	
	vtkImageData * grid = vtkImageData::New();
	
	int sx = bBoundaries ? sizeX : sizeX-2;
	int sy = bBoundaries ? sizeY : sizeY-2;
	
	grid->SetExtent(0,sx-1,0,sy-1,0,0);
	grid->SetDimensions(sx, sy, 1);
	grid->SetScalarTypeToFloat();
	grid->SetNumberOfScalarComponents(2);
	grid->AllocateScalars();
	grid->SetSpacing(1./sizeX, 1./sizeX, 1);
	grid->SetOrigin(0,0,0);
	
	int offsetX = bBoundaries ? 0 : 1;
	int offsetY = bBoundaries ? 0 : 1;
	
	for(int iy=0; iy<sy; iy++)
		for(int ix=0; ix<sx; ix++)
		{
			int index = ix+offsetX + sizeX*(iy+offsetY);
			grid->SetScalarComponentFromFloat(ix, iy, 0, 0, field1[index]);
			grid->SetScalarComponentFromFloat(ix, iy, 0, 1, field2[index]);
		}
	
	vtkXMLImageDataWriter * writer = vtkXMLImageDataWriter::New();
	writer->SetFileName(sFileName.c_str());
	writer->SetInput(grid);
	writer->Write();
	
	writer->Delete();
	grid->Delete();
#endif
}

