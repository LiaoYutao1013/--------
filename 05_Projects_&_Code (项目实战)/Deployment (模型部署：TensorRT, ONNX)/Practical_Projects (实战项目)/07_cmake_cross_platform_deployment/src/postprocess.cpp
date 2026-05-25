#include "vision/postprocess.hpp"

#include <algorithm>

namespace vision {

InferenceResult apply_threshold(const InferenceResult& result, float threshold) {
    InferenceResult filtered;
    for (std::size_t i = 0; i < result.labels.size(); ++i) {
        if (i < result.scores.size() && result.scores[i] >= threshold) {
            filtered.labels.push_back(result.labels[i]);
            filtered.scores.push_back(result.scores[i]);
        }
    }
    return filtered;
}

} // namespace vision
