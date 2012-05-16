
 
#define GID get_global_id(0)

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

inline float fill(	__read_only image2d_t vector_field,
				  const int baseX, const int baseY,
				  const float4 delta, const float4 w)
{
	const float4 tmpA = (float4)read_imagef(vector_field, sampler, (int2)(baseX, baseY));
	const float4 tmpB = (float4)read_imagef(vector_field, sampler, (int2)(baseX+1, baseY));
	
	return	dot(w,
				delta.x*			tmpA					+
				delta.y*(float4)(	tmpA.yzw,	tmpB.x)		+
				delta.z*(float4)(	tmpA.zw,	tmpB.xy)	+
				delta.w*(float4)(	tmpA.w,	    tmpB.xyz)	);
}

inline float4 computeW(const float t)
{
	return (float4)(2.f, -3.f, 3.f, -1.f)
	+ t * ((float4)(4.f, -9.5f, 8.f, -2.5f)
	+ t * ((float4)(2.5f, -7.f, 6.5f, -2.f)
	+ t * (float4)(0.5f, -1.5f, 1.5f, -0.5f)));
}

inline float4 get_delta(const int val)
{
	const int ox = val & 3;
	return (float4)(ox==0, ox==1, ox==2, ox==3);
}

__kernel void m2p_mp4(__read_only image2d_t vector_field,
					  __global float const * particle_positionsX,
					  __global float const * particle_positionsY,
					  __global float * sampled_field)
{
	if (get_global_id(0) >= NOF_ELEMENTS) return;
	
	const float2 xp = (float2)(	particle_positionsX[get_global_id(0)]*(float)SX,
							    particle_positionsY[get_global_id(0)]*(float)SX);
		
	const float2 ap = floor(xp)-1.f;
	const int iapx = (int)(ap.x);
	
	const float4 delta = get_delta(iapx);
	const float4 wx = computeW(ap.x-xp.x);
	
	const int baseX = (int)floor(iapx*0.25f);
	const int baseY = (int)(ap.y);
	
	sampled_field[get_global_id(0)] = dot((float4)(
		fill(vector_field, baseX, baseY,   delta, wx),
		fill(vector_field, baseX, baseY+1, delta, wx),
		fill(vector_field, baseX, baseY+2, delta, wx),
		fill(vector_field, baseX, baseY+3, delta, wx)),
		computeW(ap.y - xp.y));
}

__kernel void xbrunton(__read_only image2d_t vector_field,
					   __global float const * particle_positionsX,
					   __global float const * particle_positionsY,
					   __global float * sampled_field)
{
	if (GID >= NOF_ELEMENTS) return;
	
	const float2 psi = (float2)(particle_positionsX[GID],
							    particle_positionsY[GID]);
	
	const float2 xp = (float2)(	psi.x*(float)SX + (CELLSHIFT + GID % SX),
							    psi.y*(float)SX + (CELLSHIFT + GID / SX));
	
	const float2 ap = floor(xp)-1.f;
	const int iapx = (int)(ap.x);
	
	const float4 delta = get_delta(iapx);
	const float4 wx = computeW(ap.x-xp.x);
	
	const int baseX = (int)floor(iapx*0.25f);
	const int baseY = (int)(ap.y);
	
	sampled_field[GID] = dot((float4)(
		fill(vector_field, baseX, baseY,   delta, wx),
		fill(vector_field, baseX, baseY+1, delta, wx),
		fill(vector_field, baseX, baseY+2, delta, wx),
		fill(vector_field, baseX, baseY+3, delta, wx)),
		computeW(ap.y - xp.y)) + psi.x;
}

__kernel void ybrunton(__read_only image2d_t vector_field,
					          __global float const * particle_positionsX,
					          __global float const * particle_positionsY,
					          __global float * sampled_field)
{
	if (GID >= NOF_ELEMENTS) return;
	
	const float2 psi = (float2)(particle_positionsX[GID],
							    particle_positionsY[GID]);
	
	const float2 xp = (float2)(	psi.x*(float)SX + (CELLSHIFT + GID % SX),
							    psi.y*(float)SX + (CELLSHIFT + GID / SX));
	
	const float2 ap = floor(xp)-1.f;
	const int iapx = (int)(ap.x);
	
	const float4 delta = get_delta(iapx);
	const float4 wx = computeW(ap.x-xp.x);
	
	const int baseX = (int)floor(iapx*0.25f);
	const int baseY = (int)(ap.y);
	
	sampled_field[GID] = dot((float4)(
		fill(vector_field, baseX, baseY,   delta, wx),
		fill(vector_field, baseX, baseY+1, delta, wx),
		fill(vector_field, baseX, baseY+2, delta, wx),
		fill(vector_field, baseX, baseY+3, delta, wx)),
		computeW(ap.y - xp.y)) + psi.y;
}