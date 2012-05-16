/*
 *  FTLE2D.cpp
 *  FTLE2D
 *
 *  Created by Christian Conti on 1/20/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */

#include "FTLE2D.h"
#include "M2P_CPU_SSE.h"

#ifndef _DOUBLE
typedef float Real;
#else
typedef double Real;
#endif


#ifndef _LIBRARY
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
#endif


Real FTLE2D::_ComputeEigenvalue(int x, int y)
{
	const int idx = x+sizeX*y;
	const Real inv2h = .5*invh;
	
	// Compute Gradient
	const Real j00 = inv2h*(particlePos[idx+1	 ] - particlePos[idx-1	  ]);
	const Real j01 = inv2h*(particlePos[idx+sizeX] - particlePos[idx-sizeX]);
	const Real j10 = inv2h*(particlePos[idx+1+size	  ] - particlePos[idx-1+size	]);
	const Real j11 = inv2h*(particlePos[idx+sizeX+size] - particlePos[idx-sizeX+size]);
	
	// Compute Maximum Eigenvalue - formulation from maple
	const Real t1 = j10;
	const Real t2 = t1 * t1;
	const Real t3 = j11;
	const Real t4 = t3 * t3;
	const Real t5 = j00;
	const Real t6 = t5 * t5;
	const Real t7 = j01;
	const Real t8 = t7 * t7;
	const Real t9 = t2 * t2;
	const Real t16 = t4 * t4;
	const Real t21 = t6 * t6;
	const Real t24 = t8 * t8;
	const Real t29 = t9 + 2. * (t4 * t2 + t6 * t2 - t2 * t8 + t16 - t4 * t6 + t8 * t4 + t21 + t6 * t8 + t24 + 4. * t5 * t1 * t7 * t3);
	const Real t30 = sqrt(t29);
	return (t2 + t4 + t6 + t8 + t30) * .5;
}

void FTLE2D::_ComputeFTLE(const Real T)
{
	FTLEfield = new Real[size];
	Real invT = 1./fabs(T);
	
	// can't compute gradient at the boundaries
#pragma omp parallel for
	for (int iy=1; iy<sizeY-1; iy++)
		for (int ix=1; ix<sizeX-1; ix++)
		{
			// compute eigenvalue
			Real lambda = _ComputeEigenvalue(ix,iy);
			assert(lambda>=0);
			
			// compute FTLE
			FTLEfield[ix+sizeX*iy] = .5*invT*log(lambda);
		}
	
}

void FTLE2D::_InitParticlesPos()
{
#pragma omp parallel for
	for (int iy=0; iy<sizeY; iy++)
		for (int ix=0; ix<sizeX; ix++)
		{
			particlePos[ix+sizeX*iy     ] = (ix+bCentered*.5)*h;
			particlePos[ix+sizeX*iy+size] = (iy+bCentered*.5)*h;
		}
}

inline void FTLE2D::_FillMeshVel(queue<float *>& u)
{
	if ((((unsigned long)u.front()) & 0xf) != 0 ||
		(((unsigned long)meshVel) & 0xf) != 0)
	{
		cout << "Input Fields unaligned! aborting now!\n";
		abort();
	}
	
	for (int ic=0; ic<2; ic++)
	{
#pragma omp parallel for
		for (int iy=0; iy<sizeY; iy++)
			for (int ix=0; ix<sizeX; ix+=16)
			{
				_mm_store_ps(&meshVel[ix+sizeX*iy+ic*size   ],_mm_load_ps(&u.front()[ix+sizeX*iy+ic*size   ]));
				_mm_store_ps(&meshVel[ix+sizeX*iy+ic*size+ 4],_mm_load_ps(&u.front()[ix+sizeX*iy+ic*size+ 4]));
				_mm_store_ps(&meshVel[ix+sizeX*iy+ic*size+ 8],_mm_load_ps(&u.front()[ix+sizeX*iy+ic*size+ 8]));
				_mm_store_ps(&meshVel[ix+sizeX*iy+ic*size+12],_mm_load_ps(&u.front()[ix+sizeX*iy+ic*size+12]));
			}
	}
}

inline void FTLE2D::_FillFieldFromField(const float * const u, float * const output)
{
	if (bProfile) profiler.push_start("Fill");
	if ((((unsigned long)u) & 0xf) != 0 ||
		(((unsigned long)output) & 0xf) != 0)
	{
		cout << "Input Fields unaligned! aborting now!\n";
		abort();
	}
	
#pragma omp parallel
	{
#pragma omp for
		for (int i=0; i<size; i+=32)
		{
			_mm_stream_ps(&output[i   ],_mm_load_ps(&u[i   ]));
			_mm_stream_ps(&output[i+ 4],_mm_load_ps(&u[i+ 4]));
			_mm_stream_ps(&output[i+ 8],_mm_load_ps(&u[i+ 8]));
			_mm_stream_ps(&output[i+12],_mm_load_ps(&u[i+12]));
			
			_mm_stream_ps(&output[i+16],_mm_load_ps(&u[i+16]));
			_mm_stream_ps(&output[i+20],_mm_load_ps(&u[i+20]));
			_mm_stream_ps(&output[i+24],_mm_load_ps(&u[i+24]));
			_mm_stream_ps(&output[i+28],_mm_load_ps(&u[i+28]));
		}
		_mm_sfence();
	}
	
	if (bProfile) profiler.pop_stop();
}

inline void FTLE2D::_Euler2(Real dt, int ic)
{
	// SSE
	__m128 t = _mm_set_ps1(dt);
	
	float * ptrP = &particlePos[ic*size];
	float * ptrPStar = &particlePosStar[ic*size];
	const float * ptrV = &particleVel[ic*size];
	
#pragma omp parallel for
	for (int idx=0; idx<size; idx+=16)
	{
		_mm_store_ps(&ptrPStar[idx   ],_mm_add_ps(_mm_load_ps(&ptrP[idx   ]), _mm_mul_ps(t, _mm_load_ps(&ptrV[idx   ]))));
		_mm_store_ps(&ptrPStar[idx+ 4],_mm_add_ps(_mm_load_ps(&ptrP[idx+ 4]), _mm_mul_ps(t, _mm_load_ps(&ptrV[idx+ 4]))));
		_mm_store_ps(&ptrPStar[idx+ 8],_mm_add_ps(_mm_load_ps(&ptrP[idx+ 8]), _mm_mul_ps(t, _mm_load_ps(&ptrV[idx+ 8]))));
		_mm_store_ps(&ptrPStar[idx+12],_mm_add_ps(_mm_load_ps(&ptrP[idx+12]), _mm_mul_ps(t, _mm_load_ps(&ptrV[idx+12]))));
	}
}

inline void FTLE2D::_Euler3(Real dt, int ic)
{
	// SSE
	__m128 t = _mm_set_ps1(dt);
	__m128 half = _mm_set_ps1(.5);
	
	float * ptrP = &particlePos[ic*size];
	float * ptrPStar = &particlePosStar[ic*size];
	const float * ptrV = &particleVel[ic*size];
	
#pragma omp parallel for
	for (int idx=0; idx<size; idx+=16)
	{
		_mm_store_ps(&ptrP[idx   ],_mm_mul_ps(half, _mm_add_ps(_mm_load_ps(&ptrPStar[idx   ]), _mm_add_ps(_mm_load_ps(&ptrP[idx   ]), _mm_mul_ps(t, _mm_load_ps(&ptrV[idx   ]))))));
		_mm_store_ps(&ptrP[idx+ 4],_mm_mul_ps(half, _mm_add_ps(_mm_load_ps(&ptrPStar[idx+ 4]), _mm_add_ps(_mm_load_ps(&ptrP[idx+ 4]), _mm_mul_ps(t, _mm_load_ps(&ptrV[idx+ 4]))))));
		_mm_store_ps(&ptrP[idx+ 8],_mm_mul_ps(half, _mm_add_ps(_mm_load_ps(&ptrPStar[idx+ 8]), _mm_add_ps(_mm_load_ps(&ptrP[idx+ 8]), _mm_mul_ps(t, _mm_load_ps(&ptrV[idx+ 8]))))));
		_mm_store_ps(&ptrP[idx+12],_mm_mul_ps(half, _mm_add_ps(_mm_load_ps(&ptrPStar[idx+12]), _mm_add_ps(_mm_load_ps(&ptrP[idx+12]), _mm_mul_ps(t, _mm_load_ps(&ptrV[idx+12]))))));
	}
}

inline void FTLE2D::_AdvectionSubstep2(Real dt, const int channel)
{
	
	if (bProfile)
	{
		profiler.push_start("M2P");
		if (bGFLOPs) (*m2p).startTiming();
	}
	(*m2p)(meshVel, particlePos, particleVel, channel);
	if (bProfile)
	{
		if (bGFLOPs) 
		{
			(*m2p).stopTiming();
			if (!channel)
				printf("sx %d\tu%d\t%e\tGFLOP/s\n",sizeX,1,(*m2p).getPerformance());
			else
				printf("sx %d\tv%d\t%e\tGFLOP/s\n",sizeX,1,(*m2p).getPerformance());
		}
		profiler.pop_stop();
	}
	
	if (bProfile) profiler.push_start("Euler");
	_Euler2(dt,channel);
	if (bProfile) profiler.pop_stop();
}

inline void FTLE2D::_AdvectionSubstep3(Real dt, const int channel)
{
	
	if (bProfile)
	{
		profiler.push_start("M2P");
		if (bGFLOPs) (*m2p).startTiming();
	}
	(*m2p)(meshVel, particlePosStar, particleVel, channel);
	if (bProfile)
	{
		if (bGFLOPs) 
		{
			(*m2p).stopTiming();
			if (!channel)
				printf("sx %d\tu%d\t%e\tGFLOP/s\n",sizeX,1,(*m2p).getPerformance());
			else
				printf("sx %d\tv%d\t%e\tGFLOP/s\n",sizeX,1,(*m2p).getPerformance());
		}
		profiler.pop_stop();
	}
	
	if (bProfile) profiler.push_start("Euler");
	_Euler3(dt,channel);
	if (bProfile) profiler.pop_stop();
}

void FTLE2D::_TrustedAdvection(queue<Real *>& u, vector<Real>& dt, const int frame, const Real diffT)
{
	// integration with RK2
	//printf("Number of Advection Steps: %d\n", u.size());
	if (bVerbose) printf("Iterating with Trusted Advection\n");
	if (bProfile) profiler.push_start("Advect Particles");
	int i=0;
	while (u.size()>1)
	{
		_FillFieldFromField(particlePos, pos0);
		_FillFieldFromField(&particlePos[size], &pos0[size]);
		
		_FillMeshVel(u);
		
		_AdvectionSubstep2(dt[frame+i], 0);
		_AdvectionSubstep2(dt[frame+i], 1);
		
		_AdvectionSubstep3(dt[frame+i], 0);
		_AdvectionSubstep3(dt[frame+i], 1);
		
		u.pop();
		i++;
	}
	if (bVerbose) printf("Advection Completed\n");
	if (bProfile) profiler.pop_stop();
}

Real * FTLE2D::_FTLE(queue<Real *> u, vector<Real> dt, const int frame, const Real finalT)
{
	_InitParticlesPos();
	const double diffT = 0;
	
	_TrustedAdvection(u, dt, frame, diffT);

	if (bProfile) profiler.push_start("Compute FTLE");
	_ComputeFTLE(finalT);
	if (bProfile) profiler.pop_stop();
	
	return FTLEfield;
}


FTLE2D::FTLE2D(const int argc, const char ** argv, int sX, int sY, Real dh, bool bVerbose, bool bProfile) :
	parser(argc, argv), sizeX(sX), sizeY(sY), size(sX*sY), h(dh), invh(1./dh), bVerbose(bVerbose), bProfile(bProfile)
{
	bCentered = parser("-CellCentered").asBool();
	bGFLOPs = parser("-GFLOPs").asBool();
	sFilePattern = parser("-FromFile").asString();
	
	parser.save_options();
	
	assert(sizeX%16==0);
	
	printf("\n==========================\n");
	printf("Domain Size: %dx%d", sizeX, sizeY);
	printf("\n==========================\n");
	
	particlePos = new Real[sizeX*sizeY*2];
	particlePosStar = new Real[sizeX*sizeY*2];
	pos0 = new Real[sizeX*sizeY*2];
	particleVel = new Real[sizeX*sizeY*2];
	meshVel = new Real[sizeX*sizeY*2];
	FTLEfield = new Real[sizeX*sizeY];
	
	
	for (int iy=0; iy<sizeY; iy++)
		for (int ix=0; ix<sizeX; ix++)
		{
			particlePos[ix+sizeX*iy		] = (ix+bCentered*.5)*h;
			particleVel[ix+sizeX*iy		] = 0.;
			meshVel[ix+sizeX*iy			] = 0.;
			FTLEfield[ix+sizeX*iy		] = 0.;
		}
	
	for (int iy=0; iy<sizeY; iy++)
		for (int ix=0; ix<sizeX; ix++)
		{
			particlePos[ix+sizeX*iy+size] = (iy+bCentered*.5)*h;
			particleVel[ix+sizeX*iy+size] = 0.;
			meshVel[ix+sizeX*iy+size	] = 0.;
		}
	
	m2p = new OperatorM2P_CPU_SSE(sizeX, sizeY);
}

FTLE2D::~FTLE2D()
{
	delete [] particlePos;
	delete [] particlePosStar;
	delete [] pos0;
	delete [] particleVel;
	delete [] meshVel;
	delete [] FTLEfield;
	
	delete m2p;
}

// the flag bTrustDt should be set to true if the dt vector can be trusted to satisfy the CFL conditions
//	for the particles used in the advection needed in the FTLE computation
vector<Real *> FTLE2D::operator()(vector<Real *>& u, vector<Real>& dt, const Real T)
{
	// assume: time intervals between velocity fields are not "irregular"
	
	assert(u.size() == dt.size());
	assert(T>0.);
	
	vector<Real *> result;
	
	queue<Real *> u_tmp;
	Real sum_dt = 0;
	
	int e = 0;
	vector<Real *>::iterator itU = u.begin();
	vector<Real>::iterator itT = dt.begin();
	
	// prepare 1st batch of velocity fields
	do
	{
		if (e >= u.size()) break;
		
		u_tmp.push(u[e]);
		sum_dt += dt[e];
		
		e++;
	}
	while (sum_dt+1e-6 < T);
	
	int iFrameCounter = 0;
	
	bool bContinue = true;
	while (bContinue)
	{
		if (bVerbose) printf("FTLE Frame %d, u_tmp.size = %d\n", iFrameCounter, u_tmp.size());
		
		result.push_back(_FTLE(u_tmp, dt, iFrameCounter, T));
		
		sum_dt -= dt[iFrameCounter];
		u_tmp.pop();
		
		do
		{
			bContinue = !(e>=u.size());
			
			if (!bContinue) break;
			
			u_tmp.push(u[e]);
			sum_dt += dt[e];
			
			e++;
		}
		while (sum_dt+1e-6 < T);
		
		iFrameCounter++;
	}
	
	if (bProfile) profiler.printSummary();
	if (bProfile) profiler.clear();
	
	return result;
}

void FTLE2D::clear()
{
	delete [] particlePos;
	delete [] pos0;
	delete [] particleVel;
	delete [] meshVel;
	delete [] FTLEfield;
	
	particlePos = new Real[sizeX*sizeY*2];
	particleVel = new Real[sizeX*sizeY*2];
	meshVel = new Real[sizeX*sizeY*2];
	FTLEfield = new Real[sizeX*sizeY];
	
	for (int iy=0; iy<sizeY; iy++)
		for (int ix=0; ix<sizeX; ix++)
		{
			particlePos[ix+sizeX*iy		] = (ix+bCentered*.5)*h;
			particleVel[ix+sizeX*iy		] = 0.;
			meshVel[ix+sizeX*iy			] = 0.;
			FTLEfield[ix+sizeX*iy		] = 0.;
		}
	
	for (int iy=0; iy<sizeY; iy++)
		for (int ix=0; ix<sizeX; ix++)
		{
			particlePos[ix+sizeX*iy+size] = (iy+bCentered*.5)*h;
			particleVel[ix+sizeX*iy+size] = 0.;
			meshVel[ix+sizeX*iy+size	] = 0.;
		}
}
