#if !defined __TENSOR_EXPECTATION_H
#define __TENSOR_EXPECTATION_H

class TensorExpectation
{
    public:	
        torch::Tensor WTWcross(torch::Tensor &Clb, torch::Tensor &Crb, torch::Tensor &Clt, torch::Tensor &Crt, torch::Tensor &El, torch::Tensor &Eb, torch::Tensor &Er, torch::Tensor &Et, torch::Tensor &wt, const torch::Tensor &T);
        
        torch::Tensor cross(torch::Tensor &Clb, torch::Tensor &Crb, torch::Tensor &Clt, torch::Tensor &Crt, torch::Tensor &El, torch::Tensor &Eb, torch::Tensor &Er, torch::Tensor &Et, const torch::Tensor &TT);
    
	auto gateDecomposition(torch::Tensor &wlb, torch::Tensor &wnx, torch::Tensor &gate);
};

torch::Tensor TensorExpectation::WTWcross(torch::Tensor &Clb, torch::Tensor &Crb, torch::Tensor &Clt, torch::Tensor &Crt, torch::Tensor &El, torch::Tensor &Eb, torch::Tensor &Er, torch::Tensor &Et, torch::Tensor &wt, const torch::Tensor &T)
{
    torch::Tensor Tw = torch::tensordot(T, wt, {1}, {0});

    BLASINT d0 = wt.size(1)*wt.size(1);
    BLASINT d1 = wt.size(2)*wt.size(2);
    BLASINT d2 = wt.size(3)*wt.size(3);
    BLASINT d3 = wt.size(4)*wt.size(4);

    torch::Tensor wTw = torch::tensordot(torch::conj_physical(wt), Tw, {0}, {0}).permute({0, 4, 1, 5, 2, 6, 3, 7}).contiguous().view({d0, d1, d2, d3});

    return cross(Clb, Crb, Clt, Crt, El, Eb, Er, Et, wTw);
}

torch::Tensor TensorExpectation::cross(torch::Tensor &Clb, torch::Tensor &Crb, torch::Tensor &Clt, torch::Tensor &Crt, torch::Tensor &El, torch::Tensor &Eb, torch::Tensor &Er, torch::Tensor &Et, const torch::Tensor &TT)
{
    auto aal = torch::tensordot(Clt, El, {1}, {0});
    auto bbl = torch::tensordot(aal, Clb, {2}, {0});
    auto ccl = torch::tensordot(bbl, Eb, {2}, {0});

    auto aar = torch::tensordot(Crb, Er, {1}, {0});
    auto bbr = torch::tensordot(aar, Crt, {2}, {0});
    auto ccr = torch::tensordot(bbr, Et, {2}, {0});

    auto lr = torch::tensordot(ccl, ccr, {0, 3}, {3, 0});

    auto lrTT = torch::tensordot(lr, TT, {0, 1, 2, 3}, {0, 1, 2, 3}); 

    return lrTT;
}

auto TensorExpectation::gateDecomposition(torch::Tensor &wlb, torch::Tensor &wnx, torch::Tensor &gate)
{
/* wlb : w at left-bottom
*  wnx : next wave-function tensor*/

/* gate : G_{p1' p2'; p1 p2}  connecting wlb and wnx*/

    int dimp_lb = wlb.size(0);
    int dimp_nx = wnx.size(0);

    int dima_lb = wlb.size(1);
    int dimb_lb = wlb.size(2);
    int dimc_lb = wlb.size(3);
    int dimd_lb = wlb.size(4);

    int dima_nx = wnx.size(1);
    int dimb_nx = wnx.size(2);
    int dimc_nx = wnx.size(3);
    int dimd_nx = wnx.size(4);

    auto vg = gate.view({dimp_lb, dimp_nx, dimp_lb, dimp_nx}).permute({0, 2, 1, 3}).contiguous().view({dimp_lb*dimp_lb, dimp_nx*dimp_nx});

    auto [Ug, Sg, Vg] = at::linalg_svd(vg);

    auto dimS = Sg.size(0);

    auto USg = torch::einsum("b, ab->ab", {torch::sqrt(Sg), Ug}).view({dimp_lb, dimp_lb, dimS});
    auto SVg = torch::einsum("a, ab->ab", {torch::sqrt(Sg), Vg}).view({dimS, dimp_nx, dimp_nx});

    auto UW = torch::tensordot(USg, wlb, {1}, {0});
    auto WUW = torch::tensordot(torch::conj_physical(wlb), UW, {0}, {0});
    auto vWUW = WUW.permute({4, 0, 5, 1, 6, 2, 7, 3, 8}).contiguous().view({dimS, dima_lb*dima_lb, dimb_lb*dimb_lb, dimc_lb*dimc_lb, dimd_lb*dimd_lb});

    auto VW = torch::tensordot(SVg, wnx, {2}, {0});
    auto WVW = torch::tensordot(torch::conj_physical(wnx), VW, {0}, {1});
    auto vWVW = WVW.permute({4, 0, 5, 1, 6, 2, 7, 3, 8}).contiguous().view({dimS, dima_nx*dima_nx, dimb_nx*dimb_nx, dimc_nx*dimc_nx, dimd_nx*dimd_nx});

    return std::tuple<torch::Tensor, torch::Tensor>(vWUW, vWVW); // see help
}
#endif
