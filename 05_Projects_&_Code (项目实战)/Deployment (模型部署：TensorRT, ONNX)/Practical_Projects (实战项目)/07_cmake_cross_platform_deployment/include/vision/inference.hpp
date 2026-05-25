#pragma once

#include <string>
#include <vector>

namespace vision {

struct ImageInfo {
    int width;
    int height;
    std::string format;
};

struct InferenceResult {
    std::vector<std::string> labels;
    std::vector<float> scores;
};

ImageInfo load_image_info(const std::string& path);
InferenceResult run_inference(const ImageInfo& info);

} // namespace vision
