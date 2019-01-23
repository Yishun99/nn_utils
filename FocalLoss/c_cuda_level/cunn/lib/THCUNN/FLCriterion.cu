#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

template <typename T>
inline __device__ T eps();

template <>
inline __device__ float eps() { return 1e-12f; }

template <>
inline __device__ double eps() { return 1e-12; }

template <typename Dtype, typename Acctype>
struct fl_functor
{
  const Acctype alpha;
  const Acctype gamma;

  fl_functor(Acctype alpha_, Acctype gamma_)
    : alpha(alpha_), gamma(gamma_)
  {}

  template <class Tuple>
  __host__ __device__
  Acctype operator()(Tuple input)
  {
    Acctype x = ScalarConvert<Dtype, Acctype>::to(thrust::get<0>(input));
    Acctype y = ScalarConvert<Dtype, Acctype>::to(thrust::get<1>(input));
    assert(x >= 0. && x <= 1.);

    Acctype pt = x * y + (1. - x) * (1. - y);
    Acctype w = alpha * y + (1. - alpha) * (1. - y);
    w *= THCNumerics<Acctype>::pow(1. - pt, gamma);
    return -w * (y * THCNumerics<Acctype>::log(x + eps<Acctype>()) + (1. - y) * THCNumerics<Acctype>::log(1. - x + eps<Acctype>()));
  }
};

template <typename Dtype, typename Acctype>
struct fl_updateGradInput_functor
{
  const Acctype norm;
  const Acctype alpha;
  const Acctype gamma;

  fl_updateGradInput_functor(Acctype norm_, Acctype alpha_, Acctype gamma_)
    : norm(norm_), alpha(alpha_), gamma(gamma_)
  {}

  template <class Tuple>
  __host__ __device__ Dtype operator()(Tuple input)
  {
    Acctype x = ScalarConvert<Dtype, Acctype>::to(thrust::get<0>(input));
    Acctype y = ScalarConvert<Dtype, Acctype>::to(thrust::get<1>(input));

    Acctype pt = x * y + (1. - x) * (1. - y);
    Acctype alpha_factor = alpha * y + (1. - alpha) * (1. - y);
    Acctype bce_output = -(y * THCNumerics<Acctype>::log(x + eps<Acctype>()) + (1. - y) * THCNumerics<Acctype>::log(1. - x + eps<Acctype>()));

    return ScalarConvert<Acctype, Dtype>::to(
      norm * alpha_factor * THCNumerics<Acctype>::pow(1. - pt, gamma - 1.) * 
        (gamma * (1.-2.*y) * bce_output + (1.-pt) * ((x-y)/((1.-x+eps<Acctype>())*(x+eps<Acctype>())))) 
    );
  }
};


#include "generic/FLCriterion.cu"
#include "THCGenerateFloatTypes.h"
