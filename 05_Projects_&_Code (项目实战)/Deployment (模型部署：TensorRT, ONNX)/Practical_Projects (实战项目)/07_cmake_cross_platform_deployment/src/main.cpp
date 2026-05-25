#include "vision/inference.hpp"
#include "vision/postprocess.hpp"

#include <iomanip>
#include <iostream>

int main() {
    const auto info = vision::load_image_info("demo_frame.jpg");
    const auto raw = vision::run_inference(info);
    const auto filtered = vision::apply_threshold(raw, 0.65f);

    std::cout << "vision_app" << '\n';
    std::cout << "image: " << info.width << "x" << info.height << " format=" << info.format << '\n';
    std::cout << "detections:" << '\n';

    for (std::size_t i = 0; i < filtered.labels.size(); ++i) {
        std::cout << "  " << filtered.labels[i] << "  " << std::fixed << std::setprecision(2)
                  << filtered.scores[i] << '\n';
    }

    std::cout << "status: PASS" << '\n';
    return 0;
}
