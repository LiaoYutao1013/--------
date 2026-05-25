#include "vision/inference.hpp"

#include <algorithm>

namespace vision {

ImageInfo load_image_info(const std::string& path) {
    const auto has_jpg = path.find(".jpg") != std::string::npos || path.find(".jpeg") != std::string::npos;
    const auto has_png = path.find(".png") != std::string::npos;

    ImageInfo info{};
    info.width = has_png ? 640 : 1280;
    info.height = has_jpg ? 720 : 960;
    info.format = has_png ? "png" : "jpg";
    return info;
}

InferenceResult run_inference(const ImageInfo& info) {
    const float size_factor = static_cast<float>(info.width * info.height) / (1280.0f * 720.0f);

    InferenceResult result;
    result.labels = {"person", "car", "background"};
    result.scores = {
        std::min(0.95f, 0.72f + 0.10f * size_factor),
        std::min(0.93f, 0.64f + 0.08f * size_factor),
        std::min(0.40f, 0.22f + 0.04f * size_factor),
    };
    return result;
}

} // namespace vision
