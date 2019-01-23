#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/FLCriterion.cu"
#else

void THNN_(FLCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           bool sizeAverage,
           accreal alpha_,
           accreal gamma_)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_check_dim_size(state, output, 1, 0, 1);
  THCUNN_assertSameGPU(state, 2, input, target);

  ptrdiff_t size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));

  accreal sum;
  sum = thrust::transform_reduce(
    thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data)),
    thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size)),
    fl_functor<real, accreal>(alpha_, gamma_),
    (accreal) 0,
    thrust::plus<accreal>()
  );

  if (sizeAverage)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);

  THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, real>::to(sum));
}

void THNN_(FLCriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *gradInput,
           bool sizeAverage,
           accreal alpha_,
           accreal gamma_)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_assertSameGPU(state, 3, input, target, gradInput);

  ptrdiff_t size = THCTensor_(nElement)(state, input);
  accreal norm = sizeAverage ? (accreal)(1)/size : (accreal)(1);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  THCTensor_(resizeAs)(state, gradInput, input);

  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  thrust::device_ptr<real> gradInput_data(THCTensor_(data)(state, gradInput));

  thrust::transform(
    thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data)),
    thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size)),
    gradInput_data,
    fl_updateGradInput_functor<real, accreal>(norm, alpha_, gamma_)
  );

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
}

#endif
