#pragma once
#include <chrono>
#include <string>
#include <cstdio>

struct Timer {
    using clock = std::chrono::steady_clock;
    clock::time_point t0;
    std::string label;

    Timer(const std::string& lbl = "") : t0(clock::now()), label(lbl) {}

    double elapsed_ms() const {
        return std::chrono::duration<double,std::milli>(clock::now()-t0).count();
    }

    void print(const std::string& tag = "") const {
        printf("[%s%s] %.1f ms\n",
               label.empty() ? "" : (label+"::").c_str(),
               tag.c_str(), elapsed_ms());
        fflush(stdout);
    }
};
