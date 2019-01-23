TH_API void THNN_(FLCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  bool sizeAverage,
                  accreal alpha_,
                  accreal gamma_);

TH_API void THNN_(FLCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradInput,
                  bool sizeAverage,
                  accreal alpha_,
                  accreal gamma_);