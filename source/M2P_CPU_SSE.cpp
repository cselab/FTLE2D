/*
 *  M2P_CPU_SSE.cpp
 *  FTLE2D
 *
 *  Created by Christian Conti on 1/7/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */

#include "M2P_CPU_SSE.h"


#include <xmmintrin.h>
#include <emmintrin.h>
#include <limits>

#ifdef __INTEL_COMPILER
#include <smmintrin.h>
inline __m128 operator+(__m128 a, __m128 b){ return _mm_add_ps(a, b); }
inline __m128 operator&(__m128 a, __m128 b){ return _mm_and_ps(a, b); }
inline __m128 operator|(__m128 a, __m128 b){ return _mm_or_ps(a, b); }
inline __m128 operator*(__m128 a, __m128 b){ return _mm_mul_ps(a, b); }
inline __m128 operator-(__m128 a,  __m128 b){ return _mm_sub_ps(a, b); }
inline __m128d operator+(__m128d a, __m128d b){ return _mm_add_pd(a, b); }
inline __m128d operator*(__m128d a, __m128d b){ return _mm_mul_pd(a, b); }
inline __m128d operator-(__m128d a, __m128d b){ return _mm_sub_pd(a, b); }
#endif 

using namespace std;

template <int k>
inline __m128 _pack(__m128 d1, __m128 d2)  {abort(); return d1;}


template <>
inline __m128 _pack<0>(__m128 d1, __m128 d2)  { return d1; }	
template <>
inline __m128 _pack<1>(__m128 d1, __m128 d2)  { return _mm_shuffle_ps(_mm_shuffle_ps(d1, d2, _MM_SHUFFLE(0,0,2,1)),
																	  _mm_shuffle_ps(d2, d1, _MM_SHUFFLE(3,3,0,0)),
																	  _MM_SHUFFLE(0,2,1,0));}	
template <>
inline __m128 _pack<2>(__m128 d1, __m128 d2)  { return _mm_shuffle_ps(d1, d2, _MM_SHUFFLE(1,0,3,2));}	
template <>
inline __m128 _pack<3>(__m128 d1, __m128 d2)  { return _mm_shuffle_ps(_mm_shuffle_ps(d1, d2, _MM_SHUFFLE(0,0,3,3)), 
																	  _mm_shuffle_ps(d1, d2, _MM_SHUFFLE(2,1,3,3)),
																	  _MM_SHUFFLE(3,2,2,1));}
template <int c> 
inline __m128 _select(__m128 d)  {abort(); return d;}

template <> 
inline __m128 _select<0>(__m128 d)  { return _mm_shuffle_ps(d,d,  _MM_SHUFFLE(0,0,0,0));}
template <> 
inline __m128 _select<1>(__m128 d)  { return _mm_shuffle_ps(d,d,  _MM_SHUFFLE(1,1,1,1));}
template <> 
inline __m128 _select<2>(__m128 d)  { return _mm_shuffle_ps(d,d,  _MM_SHUFFLE(2,2,2,2));}
template <> 
inline __m128 _select<3>(__m128 d)  { return _mm_shuffle_ps(d,d,  _MM_SHUFFLE(3,3,3,3));}

template < int k>
inline __m128d _pack(__m128d d1, __m128d d2)  {abort(); return d1;}


template <>
inline __m128d _pack<0>(__m128d d1, __m128d d2)  { return d1; }	
template <>
inline __m128d _pack<1>(__m128d d1, __m128d d2)  { return _mm_shuffle_pd(d1, d2, _MM_SHUFFLE2(0,1));}


template <int c> 
inline __m128d _select(__m128d d1)  {abort(); return d1;}

template <> 
inline __m128d _select<0>(__m128d d1)  { return _mm_shuffle_pd(d1,d1,  _MM_SHUFFLE2(0,0));}
template <> 
inline __m128d _select<1>(__m128d d1)  { return _mm_shuffle_pd(d1,d1,  _MM_SHUFFLE2(1,1));}

#define _MM_TRANSPOSE2_PD(d1, d2)\
{	\
__m128d tmp = _mm_shuffle_pd(d1, d2, _MM_SHUFFLE2(0,0));\
d2 = _mm_shuffle_pd(d1, d2, _MM_SHUFFLE2(1,1)); \
d1 = tmp; \
}

struct M2P_workingset
{
	int sizeX, sizeY;
	__m128 invh;
	
	mutable __m128 wx0, wy0, wx1, wy1;
	mutable __m128 data0, data1, data2, data3;
	
	__m128 filtered_data0, filtered_data1, filtered_data2, filtered_data3;
	
	M2P_workingset(float h, int sizeX, int sizeY): sizeX(sizeX), sizeY(sizeY)
	{
		invh = _mm_set_ps(1/h, 1/h, 1/h, 1/h);
	}
	
	~M2P_workingset()
	{
		_mm_sfence();
	}
	
	inline void load(const float * const ptr)
	{
		data0 = _mm_loadu_ps(ptr);
		data1 = _mm_loadu_ps(ptr + sizeX);
		data2 = _mm_loadu_ps(ptr + 2*sizeX);
		data3 = _mm_loadu_ps(ptr + 3*sizeX);
	}
	
	template<int k0, int k1>
	inline void filter_setup(__m128 x, __m128 y, __m128 ax, __m128 ay)
	{	
		const __m128 a3 = _mm_set_ps(-1./2,	3./2,	-3./2,	1./2);
		const __m128 a2 = _mm_set_ps(-2,	13./2,	-7,		5./2);
		const __m128 a1 = _mm_set_ps(-5./2,	8,		-19/2.,	4);
		const __m128 a0 = _mm_set_ps(-1,	3,		-3,		2);
		
		const __m128 tx0 = _select<k0>(ax) - _select<k0>(x);
		const __m128 ty0 = _select<k0>(ay) - _select<k0>(y);
		const __m128 tx1 = _select<k1>(ax) - _select<k1>(x);
		const __m128 ty1 = _select<k1>(ay) - _select<k1>(y);
		
		{
			__m128 TMPX0, TMPY0, TMPX1, TMPY1;
			
			TMPX0 = tx0*a3;
			
			TMPX0 = TMPX0 + a2;
			TMPY0 = ty0 * a3;
			
			TMPX0 = TMPX0 * tx0;
			TMPY0 = TMPY0 + a2;
			
			TMPX0 = TMPX0 + a1;
			TMPY0 = TMPY0 * ty0;
			
			TMPX0 = TMPX0 * tx0;
			TMPY0 = TMPY0 + a1;
			
			TMPX0 = TMPX0 + a0;
			TMPY0 = TMPY0 * ty0;
			
			TMPY0 = TMPY0 + a0;
			TMPX1 = tx1*a3;
			
			TMPX1 = TMPX1 + a2;
			TMPY1 = ty1 * a3;
			
			TMPX1 = TMPX1 * tx1;
			TMPY1 = TMPY1 + a2;
			
			TMPX1 = TMPX1 + a1;
			TMPY1 = TMPY1 * ty1;
			
			TMPX1 = TMPX1 * tx1;
			TMPY1 = TMPY1 + a1;
			
			TMPX1 = TMPX1 + a0;
			TMPY1 = TMPY1 * ty1;
			
			TMPY1 = TMPY1 + a0;
			
			wx0 = TMPX0;
			wy0 = TMPY0;
			wx1 = TMPX1;
			wy1 = TMPY1;
		}
	}
	
	inline __m128 filtering(__m128 wx, __m128 wy)
	{
		return wx*(data0 * _select<0>(wy) + 
				   data1 * _select<1>(wy) + 
				   data2 * _select<2>(wy) + 
				   data3 * _select<3>(wy) );
	}
	
	void operator()(const float * const xp, const float * const yp, const float * const u_mesh, float * const u_p)
	{	
		const __m128 one = _mm_set_ps1(1);
		const __m128i one_i = _mm_set_epi32(1,1,1,1);
		
		// Boundary conditions!
		const __m128 x = _mm_min_ps(_mm_set_ps1(sizeX-3), _mm_max_ps(one, _mm_load_ps(xp)*invh));
		const __m128 y = _mm_min_ps(_mm_set_ps1(sizeY-3), _mm_max_ps(one, _mm_load_ps(yp)*invh));
		
		const __m128 apx = _mm_cvtepi32_ps(_mm_sub_epi32(_mm_cvttps_epi32(x),one_i));
		const __m128 apy = _mm_cvtepi32_ps(_mm_sub_epi32(_mm_cvttps_epi32(y),one_i));
		
		union helper
		{
			__m128 v;
			float f32[4];
		};
		
		helper offset_p;
		
		offset_p.v = _mm_add_ps(apx, _mm_mul_ps(_mm_set_ps1(sizeX), apy));
		
		//part 1
		{
			const float * const ptr0 = u_mesh + (int)offset_p.f32[0];
			const float * const ptr1 = u_mesh + (int)offset_p.f32[1];
			
			filter_setup<0,1>(x, y, apx, apy);
			
			load(ptr0);		
			filtered_data0 = filtering(wx0, wy0);
			
			load(ptr1);
			filtered_data1 = filtering(wx1, wy1);
		}
		
		//part 2
		{
			const float * const ptr2 = u_mesh + (int)offset_p.f32[2];
			const float * const ptr3 = u_mesh + (int)offset_p.f32[3];
			
			filter_setup<2,3>(x, y, apx, apy);
			
			load(ptr2);		
			filtered_data2 = filtering(wx0, wy0);
			
			load(ptr3);
			filtered_data3 = filtering(wx1, wy1);
		}
		
		_MM_TRANSPOSE4_PS(filtered_data0, filtered_data1, filtered_data2, filtered_data3); 
		_mm_store_ps(u_p, (filtered_data0 + filtered_data1) + (filtered_data2 + filtered_data3));
	}
};

struct M2P_workingset_d
{
	int sizeX, sizeY;
	__m128d invh;
	
	mutable __m128d wx0_l, wx0_h, wy0_l, wy0_h, wx1_l, wx1_h, wy1_l, wy1_h;
	mutable __m128d data0_l, data0_h, data1_l, data1_h, data2_l, data2_h, data3_l, data3_h;
	
	__m128d filtered_data0, filtered_data1; 
	
	M2P_workingset_d(double h, int sizeX, int sizeY): sizeX(sizeX), sizeY(sizeY)
	{
		invh = _mm_set_pd(1/h, 1/h);
	}
	
	~M2P_workingset_d()
	{
		_mm_sfence();
	}
	
	inline void load(const double * const ptr)
	{
		data0_l = _mm_loadu_pd(ptr);
		data0_h = _mm_loadu_pd(ptr+2);
		data1_l = _mm_loadu_pd(ptr + sizeX);
		data1_h = _mm_loadu_pd(ptr+2 + sizeX);
		data2_l = _mm_loadu_pd(ptr + 2*sizeX);
		data2_h = _mm_loadu_pd(ptr+2 + 2*sizeX);
		data3_l = _mm_loadu_pd(ptr + 3*sizeX);
		data3_h = _mm_loadu_pd(ptr+2 + 3*sizeX);
	}
	
	inline void filter_setup(__m128d x, __m128d y, __m128d ax, __m128d ay)
	{
		const __m128d a3_h = _mm_set_pd(-1./2, 3./2);
		const __m128d a3_l = _mm_set_pd(-3./2, 1./2);
		const __m128d a2_h = _mm_set_pd(-2., 13./2);
		const __m128d a2_l = _mm_set_pd(-7., 5./2);
		const __m128d a1_h = _mm_set_pd(-5./2, 8.);
		const __m128d a1_l = _mm_set_pd(-19/2., 4.);
		const __m128d a0_h = _mm_set_pd(-1, 3.);
		const __m128d a0_l = _mm_set_pd(-3, 2.);
		
		{
			const __m128d tx0 = _select<0>(ax) - _select<0>(x);
			const __m128d ty0 = _select<0>(ay) - _select<0>(y);
			
			{
				__m128d TMPX0_l, TMPY0_l;
				
				TMPX0_l = tx0 * a3_l;
				
				TMPX0_l = TMPX0_l + a2_l;
				TMPY0_l = ty0 * a3_l;
				
				TMPX0_l = TMPX0_l * tx0;
				TMPY0_l = TMPY0_l + a2_l;
				
				TMPX0_l = TMPX0_l + a1_l;
				TMPY0_l = TMPY0_l * ty0;
				
				TMPX0_l = TMPX0_l * tx0;
				TMPY0_l = TMPY0_l + a1_l;
				
				TMPX0_l = TMPX0_l + a0_l;
				TMPY0_l = TMPY0_l * ty0;
				
				TMPY0_l = TMPY0_l + a0_l;
				
				wx0_l = TMPX0_l;
				wy0_l = TMPY0_l;
				
			}
			
			{
				__m128d TMPX0_h, TMPY0_h;
				
				TMPX0_h = tx0 * a3_h;
				
				TMPX0_h = TMPX0_h + a2_h;
				TMPY0_h = ty0 * a3_h;
				
				TMPX0_h = TMPX0_h * tx0;
				TMPY0_h = TMPY0_h + a2_h;
				
				TMPX0_h = TMPX0_h + a1_h;
				TMPY0_h = TMPY0_h * ty0;
				
				TMPX0_h = TMPX0_h * tx0;
				TMPY0_h = TMPY0_h + a1_h;
				
				TMPX0_h = TMPX0_h + a0_h;
				TMPY0_h = TMPY0_h * ty0;
				
				TMPY0_h = TMPY0_h + a0_h;
				wx0_h = TMPX0_h;
				wy0_h = TMPY0_h;
			}
			
			const __m128d tx1 = _select<1>(ax) - _select<1>(x);
			const __m128d ty1 = _select<1>(ay) - _select<1>(y);
			
			{
				__m128d TMPX1_l, TMPY1_l;
				
				TMPX1_l = tx1 * a3_l;
				
				TMPX1_l = TMPX1_l + a2_l;
				TMPY1_l = ty1 * a3_l;
				
				TMPX1_l = TMPX1_l * tx1;
				TMPY1_l = TMPY1_l + a2_l;
				
				TMPX1_l = TMPX1_l + a1_l;
				TMPY1_l = TMPY1_l * ty1;
				
				TMPX1_l = TMPX1_l * tx1;
				TMPY1_l = TMPY1_l + a1_l;
				
				TMPX1_l = TMPX1_l + a0_l;
				TMPY1_l = TMPY1_l * ty1;
				
				TMPY1_l = TMPY1_l + a0_l;
				
				wx1_l = TMPX1_l;
				wy1_l = TMPY1_l;
			}
			
			{
				__m128d TMPX1_h, TMPY1_h;
				
				TMPX1_h = tx1 * a3_h;
				
				TMPX1_h = TMPX1_h + a2_h;
				TMPY1_h = ty1 * a3_h;
				
				TMPX1_h = TMPX1_h * tx1;
				TMPY1_h = TMPY1_h + a2_h;
				
				TMPX1_h = TMPX1_h + a1_h;
				TMPY1_h = TMPY1_h * ty1;
				
				TMPX1_h = TMPX1_h * tx1;
				TMPY1_h = TMPY1_h + a1_h;
				
				TMPX1_h = TMPX1_h + a0_h;
				TMPY1_h = TMPY1_h * ty1;
				
				TMPY1_h = TMPY1_h + a0_h;
				
				
				wx1_h = TMPX1_h;
				wy1_h = TMPY1_h;
			}
		}
	}
	
	inline __m128d filtering(__m128d wx_l, __m128d wx_h, __m128d wy_l, __m128d wy_h)
	{
		__m128d wy_l0 = _select<0>(wy_l);
		__m128d wy_l1 = _select<1>(wy_l);
		__m128d wy_h0 = _select<0>(wy_h);
		__m128d wy_h1 = _select<1>(wy_h);
		
		return wx_l*(data0_l * wy_l0 + 
					 data1_l * wy_l1 + 
					 data2_l * wy_h0 + 
					 data3_l * wy_h1 ) +
		wx_h*(data0_h * wy_l0 +
			  data1_h * wy_l1 +
			  data2_h * wy_h0 +
			  data3_h * wy_h1 );
	}
	
	void operator()(const double * const xp, const double * const yp, const double * const u_mesh, double * const u_p)
	{
		const __m128d one = _mm_set_pd(1,1);
		const __m128i one_i = _mm_set_epi32(1,1,1,1);
		
		const __m128d x01 = _mm_min_pd(_mm_set_pd(sizeX-3,sizeX-3), _mm_max_pd(one, _mm_load_pd(xp)*invh));
		const __m128d y01 = _mm_min_pd(_mm_set_pd(sizeY-3,sizeX-3), _mm_max_pd(one, _mm_load_pd(yp)*invh));
		
		const __m128d apx01 = _mm_cvtepi32_pd(_mm_sub_epi32(_mm_cvttpd_epi32(x01), one_i));
		const __m128d apy01 = _mm_cvtepi32_pd(_mm_sub_epi32(_mm_cvttpd_epi32(y01), one_i));
		
		union helper
		{
			__m128d v;
			double f64[2];
		};
		
		helper offset_p;
		
		offset_p.v = _mm_add_pd(apx01, _mm_mul_pd(_mm_set_pd(sizeX,sizeX), apy01));
		
		const double * const ptr0 = u_mesh + (int)offset_p.f64[0];
		const double * const ptr1 = u_mesh + (int)offset_p.f64[1];
		
		filter_setup(x01, y01, apx01, apy01);
		
		load(ptr0);	
		filtered_data0 = filtering(wx0_l, wx0_h, wy0_l, wy0_h);
		
		load(ptr1);
		filtered_data1 = filtering(wx1_l, wx1_h, wy1_l, wy1_h);
		
		_MM_TRANSPOSE2_PD(filtered_data0, filtered_data1);
		
		_mm_store_pd(u_p, filtered_data0 + filtered_data1);
	}
};

void OperatorM2P_CPU_SSE::_m2p_super_aggressive(int sizeX, int sizeY, const float * const x_p, const float * const y_p, const float * const u_mesh, float * const u_p)
{
	if (((unsigned long)x_p) & 0xf != 0 ||
		((unsigned long)y_p) & 0xf != 0 ||
		((unsigned long)u_mesh) & 0xf != 0 ||
		((unsigned long)u_p) & 0xf != 0 ||
		sizeX % 16 != 0 || sizeX < 16)
	{
		printf("((unsigned long)x_p) & 0xf = %d\n",((unsigned long)x_p) & 0xf);
		printf("((unsigned long)y_p) & 0xf = %d\n",((unsigned long)y_p) & 0xf);
		printf("((unsigned long)u_mesh) & 0xf = %d\n",((unsigned long)u_mesh) & 0xf);
		printf("((unsigned long)u_p) & 0xf = %d\n",((unsigned long)u_p) & 0xf);
		printf("sizeX = %d\n",sizeX);
		abort();
	}
	
	
#pragma omp parallel for
	for(int iy=0; iy<sizeY; iy++)
	{
		M2P_workingset m2p_workitem(1.f/sizeX, sizeX, sizeY);
		
		const int offset = iy*sizeX;
		
		for(int ix=0; ix<sizeX; ix+=16)
		{
			m2p_workitem(x_p + offset + ix,		 y_p + offset + ix,		 u_mesh, u_p + ix + offset);
			m2p_workitem(x_p + offset + ix + 4,  y_p + offset + ix + 4,	 u_mesh, u_p + ix + offset + 4);
			m2p_workitem(x_p + offset + ix + 8,  y_p + offset + ix + 8,	 u_mesh, u_p + ix + offset + 8);
			m2p_workitem(x_p + offset + ix + 12, y_p + offset + ix + 12, u_mesh, u_p + ix + offset + 12);
		}
	}
}

void OperatorM2P_CPU_SSE::_m2p_super_aggressive(int sizeX, int sizeY, const double * const x_p, const double * const y_p, const double * const u_mesh, double * const u_p)
{
	if (((unsigned long)x_p) & 0xf != 0 ||
		((unsigned long)y_p) & 0xf != 0 ||
		((unsigned long)u_mesh) & 0xf != 0 ||
		((unsigned long)u_p) & 0xf != 0 ||
		sizeX % 16 != 0 || sizeX < 16)
	{
		std::cout << "Data for M2P is unaligned!\n";
		abort();
	}
	
#pragma omp parallel for
	for(int iy=0; iy<sizeY; iy++)
	{
		M2P_workingset_d m2p_workitem(1./sizeX, sizeX, sizeY);
		
		const int offset = iy*sizeX;
		
		for(int ix=0; ix<sizeX; ix+=8)
		{
			m2p_workitem(x_p + offset + ix,		y_p + offset + ix,	   u_mesh, u_p + ix + offset);
			m2p_workitem(x_p + offset + ix + 2, y_p + offset + ix + 2, u_mesh, u_p + ix + offset + 2);
			m2p_workitem(x_p + offset + ix + 4, y_p + offset + ix + 4, u_mesh, u_p + ix + offset + 4);
			m2p_workitem(x_p + offset + ix + 6, y_p + offset + ix + 6, u_mesh, u_p + ix + offset + 6);
		}
	}
}

OperatorM2P_CPU_SSE::OperatorM2P_CPU_SSE(int sX, int sY) : OperatorM2P(sX,sY) {}

inline void OperatorM2P_CPU_SSE::operator()(const Real * const u_m, const Real * const x_p, Real * u_p, const int channel)
{
	// NEVER, EVER set the output u_p with the input u_m!!!
	assert(u_m!=u_p);
	_m2p_super_aggressive(sizeX, sizeY, &x_p[0], &x_p[sizeX*sizeY], &u_m[channel*sizeX*sizeY], &u_p[channel*sizeX*sizeY]);
}

inline void OperatorM2P_CPU_SSE::operator()(const Real * const u_m, const Real * const x_p, Real * u_p)
{
	assert(u_m!=u_p);
	_m2p_super_aggressive(sizeX, sizeY, &x_p[0], &x_p[sizeX*sizeY],&u_m[0], &u_p[0]);
	_m2p_super_aggressive(sizeX, sizeY, &x_p[0], &x_p[sizeX*sizeY],&u_m[sizeX*sizeY], &u_p[sizeX*sizeY]);
}

inline void OperatorM2P_CPU_SSE::startTiming()
{
	timer.start();
}

inline void OperatorM2P_CPU_SSE::stopTiming()
{
	time = timer.stop();
}

inline double OperatorM2P_CPU_SSE::getBandwidth()
{
	// memory traffic: average of estimated best and worst values
	return (5+17+2)*sizeof(Real)*sizeX*sizeY/1024./1024./1024./time;
}