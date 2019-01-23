TH_API void THNN_(FLCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          accreal alpha_,
          accreal gamma_);
TH_API void THNN_(FLCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          accreal alpha_,
          accreal gamma_);