#include "globalconfig.h"
#include "stringname.h"

/*
C  ---- 0    C  1 ----     C  0           C    1      E  ---- 0     E  2 ----    E   0  1  2    E  -------
   |                 |        |                |         |--- 1        1 ---|        |  |  |       |  |  |
   1                 0        ---- 1      0 ----         ---- 2        0 ----        -------       2  1  0

*/

#if !defined __CORNER_CONTRACTION_H
#define __CORNER_CONTRACTION_H

class CornerContraction
{
    public:
 /*'s' means Cts, Els, Ers, Ets, Ebs are for the site of 'T'*/

        torch::Tensor LTContraction(const torch::Tensor &Cts, const torch::Tensor &Els, const torch::Tensor &Ets, const torch::Tensor &T);
        torch::Tensor RTContraction(const torch::Tensor &Cts, const torch::Tensor &Ers, const torch::Tensor &Ets, const torch::Tensor &T);
        torch::Tensor LBContraction(const torch::Tensor &Cts, const torch::Tensor &Els, const torch::Tensor &Ebs, const torch::Tensor &T);
        torch::Tensor RBContraction(const torch::Tensor &Cts, const torch::Tensor &Ers, const torch::Tensor &Ebs, const torch::Tensor &T);
};

torch::Tensor CornerContraction::LTContraction(const torch::Tensor &Cts, const torch::Tensor &Els, const torch::Tensor &Ets, const torch::Tensor &T)
{
#ifdef DEBUG_CTT_TIME
    auto timer_start = std::chrono::high_resolution_clock::now();
#endif
	
    auto dim0 = T.size(2) * Ets.size(0);  // from bottom to top
    auto dim1 = Els.size(2) * T.size(1);  // from left to right
    auto ec = torch::tensordot(Ets, Cts, {2}, {0});
    auto ece = torch::tensordot(ec, Els, {2}, {0});
    auto ltc = torch::tensordot(ece, T, {2, 1}, {0, 3}).permute({3, 0, 1, 2}).contiguous().view({dim0, dim1});

#ifdef DEBUG_CTT_TIME
    auto timer_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start);

    std::cout << "LTContraction time is " << duration.count() << " ms\n";
#endif    

    return ltc;
}

torch::Tensor CornerContraction::RTContraction(const torch::Tensor &Cts, const torch::Tensor &Ers, const torch::Tensor &Ets, const torch::Tensor &T)
{
#ifdef DEBUG_CTT_TIME
    auto timer_start = std::chrono::high_resolution_clock::now();
#endif
	
    auto dim0 = T.size(1) * Ers.size(0);  // left to right
    auto dim1 = T.size(0) * Ets.size(2);  // bottom to top
    auto ec = torch::tensordot(Ers, Cts, {2}, {0});
    auto ece = torch::tensordot(ec, Ets, {2}, {0});
    auto rtc = torch::tensordot(T, ece, {2, 3}, {1, 2}).permute({1, 2, 0, 3}).contiguous().view({dim0, dim1});

#ifdef DEBUG_CTT_TIME
    auto timer_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start);

    std::cout << "RTContraction time is " << duration.count() << " ms\n";
#endif
    
    return rtc;
}

torch::Tensor CornerContraction::LBContraction(const torch::Tensor &Cts, const torch::Tensor &Els, const torch::Tensor &Ebs, const torch::Tensor &T)
{
#ifdef DEBUG_CTT_TIME
    auto timer_start = std::chrono::high_resolution_clock::now();
#endif

    auto dim0 = Els.size(0) * T.size(3);   // left to right
    auto dim1 = Ebs.size(2) * T.size(2);   // bottom to top
    auto ec = torch::tensordot(Els, Cts, {2}, {0});
    auto ece = torch::tensordot(ec, Ebs, {2}, {0});
    auto lbc = torch::tensordot(ece, T, {1, 2}, {0, 1}).permute({0, 3, 1, 2}).contiguous().view({dim0, dim1});

#ifdef DEBUG_CTT_TIME
    auto timer_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start);

    std::cout << "LBContraction time is " << duration.count() << " ms\n";
#endif    

    return lbc;
}

torch::Tensor CornerContraction::RBContraction(const torch::Tensor &Cts, const torch::Tensor &Ers, const torch::Tensor &Ebs, const torch::Tensor &T)
{
#ifdef DEBUG_CTT_TIME
    auto timer_start = std::chrono::high_resolution_clock::now();
#endif
	
    auto dim0 = Ebs.size(0) * T.size(0);    // bottom to top
    auto dim1 = T.size(3) * Ers.size(2);    // left to right
    auto ec = torch::tensordot(Ebs, Cts, {2}, {0});
    auto ece = torch::tensordot(ec, Ers, {2}, {0});
    auto rbc = torch::tensordot(T, ece, {1, 2}, {1, 2}).permute({2, 0, 1, 3}).contiguous().view({dim0, dim1});

#ifdef DEBUG_CTT_TIME
    auto timer_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start);

    std::cout << "RBContraction time is " << duration.count() << " ms\n";
#endif    

    return rbc;
}

#endif
