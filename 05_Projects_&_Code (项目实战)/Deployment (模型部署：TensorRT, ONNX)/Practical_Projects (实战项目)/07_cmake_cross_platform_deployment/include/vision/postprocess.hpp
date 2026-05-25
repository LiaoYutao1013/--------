#pragma once

#include "vision/inference.hpp"

namespace vision {

InferenceResult apply_threshold(const InferenceResult& result, float threshold);

} // namespace vision
