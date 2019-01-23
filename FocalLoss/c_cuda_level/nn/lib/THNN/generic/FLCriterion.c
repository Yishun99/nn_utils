#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/FLCriterion.c"
#else

#define EPS 1e-12

void THNN_(FLCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          accreal alpha_,
          accreal gamma_)
{
  THNN_CHECK_NELEMENT(input, target);
  THNN_CHECK_DIM_SIZE(output, 1, 0, 1);

  real alpha = TH_CONVERT_ACCREAL_TO_REAL(alpha_);
  real gamma = TH_CONVERT_ACCREAL_TO_REAL(gamma_);
  real sum = 0;

  TH_TENSOR_APPLY2(real, input, real, target,
    real x = *input_data;
    real y = *target_data;
    THAssertMsg(x >= 0. && x <= 1.,
      "input value should be between 0~1, but got %f",
    (double) x);

    real pt = x * y + (1. - x) * (1. - y);
    real w = alpha * y + (1. - alpha) * (1. - y);
    w *= pow(1. - pt, gamma);
    sum -= w * (y * log(x + EPS) + (1. - y) * log(1. - x + EPS));
  );

  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(FLCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          accreal alpha_,
          accreal gamma_)
{
  THNN_CHECK_NELEMENT(input, target);

  real alpha = TH_CONVERT_ACCREAL_TO_REAL(alpha_);
  real gamma = TH_CONVERT_ACCREAL_TO_REAL(gamma_);
  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);

  THTensor_(resizeAs)(gradInput, input);

  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
    real x = *input_data;
    real y = *target_data;

    real pt = x * y + (1. - x) * (1. - y);
    real alpha_factor = alpha * y + (1. - alpha) * (1. - y);
    real bce_output = -(y * log(x + EPS) + (1. - y) * log(1. - x + EPS));
    *gradInput_data = norm * alpha_factor * pow(1.-pt, gamma-1.) * 
      (gamma * (1.-2.*y) * bce_output + (1.-pt) * ((x - y) / ((1.-x+EPS) * (x+EPS))));
  );
}

#undef EPS

#endif
