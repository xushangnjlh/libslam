#pragma once

#include <chrono>
#include <cstdlib>
#include <ctime>
class TicToc
{
public:
    TicToc()
    {
        tic();
    }
    void tic()
    {
        start = std::chrono::steady_clock::now();
    }
    double toc()
    {
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapse = std::chrono::duration_cast<std::chrono::duration<double> >(end - start);
        return elapse.count()*1000;
    }
private:
    std::chrono::steady_clock::time_point start, end;
};