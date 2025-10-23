#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <cassert>
#include <random>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "iPEPS.h"
#include "localconfig.h"
#include "optimizationLocal.h"

using namespace std;
using namespace chrono;

constexpr double INITIAL_BEST_ENERGY = 999999.0;
constexpr double GMAX = 10.0;
constexpr int ELQ_ITERNUM = 10; // [修改] 原为 1000
constexpr int STAGNATION_THRESHOLD = 50;
constexpr double EXPLORER_RATIO = 0.6;
//constexpr double MIN_SCALE = 0.03;

constexpr int WarmNum = 5; // [修改] 原为 200

constexpr double PROG_COEFF = 0.99;

class ParallelRandom 
{
    std::vector<mt19937> generators;
    uniform_real_distribution<double> dist{0.0, 1.0};
    normal_distribution<double> gauss_dist{0.0, 1.0};
public:
    ParallelRandom() 
    {
        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#endif
        random_device rd;
        generators.reserve(num_threads);
        for (int i = 0; i < num_threads; ++i) generators.emplace_back(rd());
    }

    double operator()() 
    {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        return dist(generators[thread_id]);
    }

    double gaussian() 
    {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        return gauss_dist(generators[thread_id]);
    }

    double levy(double lvbeta = 1.5) 
    {
        double sigma_u = pow((tgamma(1 + lvbeta) * sin(M_PI * lvbeta / 2)) / (tgamma((1 + lvbeta) / 2) * lvbeta * pow(2, (lvbeta - 1) / 2)), 1.0 / lvbeta);
        double u = gaussian() * sigma_u;
        double v = gaussian();
        double eps = 1e-7;
        v = (abs(v) < eps) ? ((v >= 0) ? eps : -eps) : v;
	//return u / pow(abs(v), 1.0 / lvbeta);
	return std::clamp(u / pow(abs(v), 1.0 / lvbeta), -10.0, 10.0);
    }
};

struct ELQParticle 
{
    std::vector<torch::Tensor> positions;      // The tensors may be different in some models, so here vector is the best choice.
    std::vector<torch::Tensor> best_positions;
    torch::Tensor best_energy;
    int stagnation_count = 0;
    bool is_explorer = true;
    double search_scale = 1.0;
    double scale_decay = 0.99;

    int gidx;    // save the index of the group and position of a particle
};

double multi_scale_perturbation(int iter, double base_scale, double gBest_improvement, double progress) 
{
    double scale = base_scale * (1.0 - 0.8 * progress);
    if (gBest_improvement < 1e-6) scale = min(0.02, scale * 1.1);
    else scale = max(0.01, scale * 0.98);
    scale *= (0.99 + 0.1 * sin(iter * 0.1));
    return scale;
}

double temperature_schedule(int iter, double initial_temp, double current_temp, double final_temp, int max_iter)
{
    //double progress =  PROG_COEFF * iter / static_cast<double>(max_iter);
    //return final_temp + (initial_temp - final_temp) * pow(1.0 - progress, 3.0);

    if (iter < WarmNum) return current_temp;	
    else return std::max(final_temp, current_temp*0.995);	
}

torch::Tensor flatten(const std::vector<torch::Tensor> &tensors)
{
    torch::Tensor stacked = torch::stack(tensors).contiguous();
    return stacked.view({stacked.numel()});
}

std::vector<torch::Tensor> unflatten(const torch::Tensor &flat)
{ 
    auto reshaped = flat.reshape({LY*LX, dims, DB, DB, DB, DB});
    auto split = torch::tensor_split(reshaped, LY*LX, 0);
    std::vector<torch::Tensor> ts;
    for (auto& t : split) ts.push_back(t.squeeze(0));
    return ts;
}

void construct_mBest_with_diversity(int sn, int elite_count, const std::vector<pair<double, int>>&energy_with_index_sn, const std::vector<ELQParticle>& elqparticles, std::vector<torch::Tensor>& mBestPosSn, double progress)
{
    // === 距离度量：扁平化张量的 L2 距离 ===
    auto tensor_distance = [](const std::vector<torch::Tensor>& a, const std::vector<torch::Tensor>& b) 
    {
        auto flat_a = flatten(a);
        auto flat_b = flatten(b);
   
        auto norm_a = torch::norm(flat_a);
	auto norm_b = torch::norm(flat_b);

        flat_a = flat_a/norm_a;
	flat_b = flat_b/norm_b;

	// 计算两种可能的相位（原始和反相）
        double dist1 = torch::norm(flat_a - flat_b).item<double>();
        double dist2 = torch::norm(flat_a + flat_b).item<double>(); // 考虑全局相位翻转

        // 取最小值，因为两种相位表示相同的物理状态
        return std::min(dist1, dist2);
    };

    // === 候选集合：取能量最小的 2 * elite_count 个粒子 ===
    int candidate_count = min((int)energy_with_index_sn.size(), 2 * elite_count);
    std::vector<int> candidate_indices(candidate_count);
    for (int k = 0; k < candidate_count; ++k) candidate_indices[k] = energy_with_index_sn[k].second;

    // === 多样性选择：贪心选出 elite_count 个互距较远的粒子 ===
    std::vector<int> selected_indices;
    selected_indices.push_back(candidate_indices[0]); // 先选能量最小的

    while ((int)selected_indices.size() < elite_count)
    {
        double max_min_dist = -1.0;
        int best_candidate = -1;

        for (int idx : candidate_indices)
        {
            if (find(selected_indices.begin(), selected_indices.end(), idx) != selected_indices.end()) continue;

            double min_dist = 1e100;
            for (int sel : selected_indices)
            {
                double dist = tensor_distance(elqparticles[idx].best_positions, elqparticles[sel].best_positions);
                min_dist = min(min_dist, dist);
            }

            if (min_dist > max_min_dist)
            {
                max_min_dist = min_dist;
                best_candidate = idx;
            }
        }

        if (best_candidate >= 0) selected_indices.push_back(best_candidate);
        else break;
    }

    // === Softmax 权重计算（数值稳定） ===
    double alpha = 10.0 + 30.0 * progress;
    double min_energy = elqparticles[selected_indices[0]].best_energy.item<double>();

    std::vector<double> weights(selected_indices.size());
    double weight_sum = 0.0;
    for (size_t k = 0; k < selected_indices.size(); ++k)
    {
        int i = selected_indices[k];
        double energy = elqparticles[i].best_energy.item<double>();
        weights[k] = exp(-alpha * (energy - min_energy));  // 数值稳定
        weight_sum += weights[k];
    }
    for (double& w : weights) w /= weight_sum;

    // === 构造 mBestPosSn[j] ===
    for (int j = 0; j < LY * LX; ++j)
    {
        mBestPosSn[j] = torch::zeros_like(elqparticles[0].positions[j]);
        for (size_t k = 0; k < selected_indices.size(); ++k)
        {
            int i = selected_indices[k];
            mBestPosSn[j] += weights[k] * elqparticles[i].best_positions[j];
        }
    }
}

void construct_mBest_without_diversity(int sn, int elite_count, const std::vector<pair<double, int>>& energy_with_index_sn, const std::vector<ELQParticle>& elqparticles, std::vector<torch::Tensor>& mBestPosSn, double progress)
{
    std::vector<double> weights(elite_count);
    double weight_sum = 0.0;

    // 获取最小能量（注意 energy_with_index_sn 已按升序排列）
    double min_energy = elqparticles[energy_with_index_sn[0].second].best_energy.item<double>();

    double alpha = 10.0 + 30.0 * progress;

    for (int k = 0; k < elite_count; ++k)
    {
        int i = energy_with_index_sn[k].second;
        double energy = elqparticles[i].best_energy.item<double>();
        weights[k] = exp(-alpha * (energy - min_energy));
        weight_sum += weights[k];
    }

    for (int k = 0; k < elite_count; ++k) weights[k] /= weight_sum;

    // Softmax 加权平均构造 mBestPos[sn]
    for (int j = 0; j < LY*LX; j++)
    {
        mBestPosSn[j] = torch::zeros_like(elqparticles[0].best_positions[j]);

        for (int k = 0; k < elite_count; ++k)
        {
            int i = energy_with_index_sn[k].second;
            mBestPosSn[j] += weights[k] * elqparticles[i].best_positions[j];
        }
    }
}

struct ChaosState 
{
    double x;    // current chaotic variable in (0,1)
    double mu;   // logistic map parameter, e.g. 3.99
    double eps;  // base perturbation strength (0..0.1)
};

inline double logistic_map_step(double x, double mu = 3.99) 
{
    x = mu * x * (1.0 - x);
    // 防止数值退化（极端情况）
    if (x <= 1e-12 || x >= 1.0 - 1e-12) x = 0.6180339887498948 * (0.5); // 黄金比例相关初值微调
    return x;
}

void chaoticPerturbateGBest(std::vector<torch::Tensor> &gbestpos, ChaosState &cs, const int &rejfreq, const double &progress, const double &gmax) 
{
    // 更新混沌序列
    cs.x = logistic_map_step(cs.x, cs.mu);
    // 映成 [-1,1]
    double fac = 2.0 * cs.x - 1.0;

    const double stagnation_factor = 1.0 + 0.2 * std::min(rejfreq, 10);  // 停滞越久扰动越大
    double scale = cs.eps * (1.0 - 0.95 * progress) * stagnation_factor;
    scale = std::min(scale, 0.2); // 上限避免过大扰动

    for (auto &t : gbestpos) 
    {
	if (drand48() < 0.3)
	{	
            // 保证 dtype 为 double
            auto noise = torch::rand_like(t, torch::TensorOptions().dtype(torch::kF64));
            // noise ∈ [0,1) -> 映射到 [-1,1)
            noise = (noise * 2.0 - 1.0);
            // 将 fac 作为序列偏向乘子, 并用 GMAX 做幅度尺度
            t.add_(noise * (scale * fac) * gmax).clamp_(-gmax, gmax);
	}
    }
}

struct SubswarmsTp
{
    double initialTp;
    double currentTp;
    double finalTp;   

    double levy_scale;

    int explorer_count;
};

int main(int argc, char* argv[]) 
{
    assert(argc == 2);
    ofstream coutfile("info.txt", ios::app);
    auto coutbuff = cout.rdbuf();
    cout.rdbuf(coutfile.rdbuf());

#ifdef _OPENMP
    omp_set_num_threads(THREADS_NUM);
    cout << "Max threads: " << omp_get_max_threads() << endl;

    omp_set_nested(1);
    omp_set_max_active_levels(2);
#endif

    auto timer_start = high_resolution_clock::now();
    const BLASINT chi = KAI;
    auto tsoptions = torch::TensorOptions().dtype(torch::kF64);

    assert(PSONUM%SUBSWARMS == 0);

    ParallelRandom rand_gen;
    std::vector<ELQParticle> elqparticles(PSONUM);

    int mapSS2P[PSONUM];

    for (int i = 0; i < PSONUM; ++i) 
    {
        auto& p = elqparticles[i];
        p.positions.resize(LY * LX);
        p.best_positions.resize(LY * LX);
        p.best_energy = torch::tensor({INITIAL_BEST_ENERGY}, torch::kF64);
        p.is_explorer = true;
        p.search_scale = p.is_explorer ? 1.0 : 0.5;

        for (int yx = 0; yx < LY * LX; ++yx) 
	{
            int PEPSINIT = static_cast<int>(rand_gen() * 6)%6;
            if (PEPSINIT == 0) p.positions[yx] = torch::rand({dims, DB, DB, DB, DB}, tsoptions);
            else if (PEPSINIT == 1) p.positions[yx] = torch::rand({dims, DB, DB, DB, DB}, tsoptions) - 0.5;
            else if (PEPSINIT == 2) p.positions[yx] = torch::rand({dims, DB, DB, DB, DB}, tsoptions) - torch::rand({dims, DB, DB, DB, DB}, tsoptions);
            else if (PEPSINIT == 3) p.positions[yx] = torch::randn({dims, DB, DB, DB, DB}, tsoptions);
            else if (PEPSINIT == 4) p.positions[yx] = torch::randn({dims, DB, DB, DB, DB}, tsoptions) - 0.5;
            else if (PEPSINIT == 5) p.positions[yx] = torch::randn({dims, DB, DB, DB, DB}, tsoptions) - torch::randn({dims, DB, DB, DB, DB}, tsoptions);
            else p.positions[yx] = torch::randn({dims, DB, DB, DB, DB}, tsoptions) - torch::rand({dims, DB, DB, DB, DB}, tsoptions);

            if (p.is_explorer) p.positions[yx] = 0.5 * GMAX * p.positions[yx];
            else p.positions[yx] = 0.2 * GMAX * p.positions[yx];
        }

	p.gidx = i;

	mapSS2P[p.gidx] = i;
    }

    std::vector<SubswarmsTp> sst(SUBSWARMS);

    for (int sn = 0; sn < SUBSWARMS; sn++)
    {
	sst[sn].initialTp = 200 + sn * 200/SUBSWARMS;
	sst[sn].currentTp = sst[sn].initialTp;
	sst[sn].finalTp   = 0.1 + sn * 0.1;

	sst[sn].levy_scale = 0.04 * pow(1.1, sn);

	sst[sn].explorer_count = GPNUM;
    } 

    std::vector<ChaosState> chaos(SUBSWARMS);
    for (int sn = 0; sn < SUBSWARMS; ++sn) 
    {
        chaos[sn].x = 0.6180339887 * (0.5 + 0.01 * sn); // 选一个不一样的初值，避免同步
        chaos[sn].mu = 3.99;
        // 初始扰动强度，可按子群略有差异
        chaos[sn].eps = 0.02 * (1.0 - 0.3 * static_cast<double>(sn)/SUBSWARMS);
    }

    char filename[120];

    std::vector<torch::Tensor> gBestPos[SUBSWARMS];
    torch::Tensor gBest[SUBSWARMS];
    double prev_gBest[SUBSWARMS];

    std::vector<torch::Tensor> mBestPos[SUBSWARMS];

    for (int sn = 0; sn < SUBSWARMS; sn++)
    {
        gBestPos[sn].resize(LY*LX);	

        gBest[sn] = torch::tensor({INITIAL_BEST_ENERGY}, torch::kF64);

	prev_gBest[sn] = gBest[sn].item<double>();

	mBestPos[sn].resize(LY*LX);
    }

    std::vector<double> exweight(GPNUM);
    for (int i = 0; i < GPNUM; i++) exweight[i] = std::pow(3.0, i);
    double sum = std::accumulate(exweight.begin(), exweight.end(), 0.0);
    for (auto &x : exweight) x /= sum;
    for (int i = 1; i < GPNUM; i++) exweight[i] += exweight[i-1];
    exweight[GPNUM-1] += 1.0e-10;
    for (auto &x : exweight) std::cout << x << "  ";
    std::cout << std::endl;


    #pragma omp parallel for schedule(dynamic) 
    for (int i = 0; i < PSONUM; ++i) 
    {
	int sn = elqparticles[i].gidx/GPNUM;  // group index

	std::vector<torch::Tensor> vec;
	for (auto &p : elqparticles[i].positions) vec.push_back(p.to(torch::kFloat));
        iPEPS sq(vec);
        auto energy = sq.groundEnergy(i, 'I', chi, {1.0e-2, 1.0e-1}, true, false, true, true).to(torch::kF64);
        elqparticles[i].best_energy = energy.clone();
        for (int j = 0; j < LY * LX; ++j) elqparticles[i].best_positions[j] = elqparticles[i].positions[j].clone();
        
        #pragma omp critical
        if (energy.item<double>() < gBest[sn].item<double>()) 
	{
            gBest[sn] = energy.clone();
            for (int j = 0; j < LY * LX; ++j) gBestPos[sn][j] = elqparticles[i].positions[j].clone(); 
        }
    }

    int maxLBFGSiter = 5;

    std::vector<pair<double, int>> energy_with_index[SUBSWARMS];
    for (int sn = 0; sn < SUBSWARMS; sn++) energy_with_index[sn].resize(GPNUM);

    int rejectedfreq[SUBSWARMS];
    for (int i = 0; i < SUBSWARMS; i++) rejectedfreq[i] = 0;

    for (int iter = 0; iter < ELQ_ITERNUM + WarmNum; ++iter) 
    {
        auto timer_start = std::chrono::high_resolution_clock::now();

        double progress = iter <= WarmNum ? 0.0 : (iter - WarmNum)/static_cast<double>(ELQ_ITERNUM);
        double global_scale = max(0.1, 1.0 - 0.9 * progress);
        double beta = 0.3 + 0.68 * pow(1.0 - progress, 2.0);  // 区间：0.98 -> 0.3

        double elite_ratio = 0.4 - 0.3 * progress;  // 后期降到10%
        int elite_count = max(1, static_cast<int>(GPNUM * elite_ratio));

        #pragma omp parallel for schedule(dynamic) 
        for (int sn = 0; sn < SUBSWARMS; sn++)
	{
	    energy_with_index[sn].resize(GPNUM);

            for (int ip = 0; ip < GPNUM; ++ip)
            {
                int gidx = sn * GPNUM + ip;
                int i = mapSS2P[gidx];
                energy_with_index[sn][ip] = {elqparticles[i].best_energy.item<double>(), i};
            }

            sort(energy_with_index[sn].begin(), energy_with_index[sn].end());

	    if (iter < WarmNum + ELQ_ITERNUM/3) construct_mBest_with_diversity(sn, GPNUM, energy_with_index[sn], elqparticles, mBestPos[sn], progress);
            else construct_mBest_without_diversity(sn, elite_count, energy_with_index[sn], elqparticles, mBestPos[sn], progress);

            sst[sn].currentTp = temperature_schedule(iter, sst[sn].initialTp, sst[sn].currentTp, sst[sn].finalTp, ELQ_ITERNUM);
        }

        std::cout << "Current T in iteration " << iter << " is " << sst[0].currentTp << std::endl;

//        auto timer_temp = std::chrono::high_resolution_clock::now();
//        auto duration_temp = std::chrono::duration_cast<std::chrono::milliseconds>(timer_temp - timer_start);
//        std::cout << "Iteration "<<iter <<" temperature completed in " << duration_temp.count()/1000.0 << " S\n" << std::endl;

        bool printable = false;
	if (iter%100 == 0) printable = true;

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < PSONUM; ++i) 
	{
	    int sn = elqparticles[i].gidx/GPNUM;
            int ip = elqparticles[i].gidx%GPNUM;

            double lvy_scale = multi_scale_perturbation(iter, sst[sn].levy_scale, abs(prev_gBest[sn] - gBest[sn].item<double>()), progress);

	    double currentTp = sst[sn].currentTp;

	    std::vector<torch::Tensor> new_pos(LY * LX);
            for (int j = 0; j < LY * LX; ++j) 
	    {
                /*Warning: center may significantly change the convergence speed of the algorithm.*/

                //auto center = (elqparticles[i].best_positions[j] + gBestPos[j] + mBestPos[j]) / 3.0;
                //auto center = 0.2 * elqparticles[i].best_positions[j] + 0.4 * gBestPos[j] + 0.4 * mBestPos[j];

		double p = pow(progress, 2.0);

		auto center = 0.75 * (1.0 - p) * elqparticles[i].best_positions[j] + 0.2 * (1.0 - p) * mBestPos[sn][j] + (0.05 + 0.95 * p) * gBestPos[sn][j];
                //if (iter <= WarmNum + ELQ_ITERNUM * 0.1) center = (0.8 - 0.1 * rand_gen())* elqparticles[i].best_positions[j] + (0.15 + 0.1 * rand_gen()) * mBestPos[sn][j] + 0.05 * torch::randn_like(elqparticles[i].best_positions[j]);
                if (iter <= WarmNum + ELQ_ITERNUM * 0.1) center = (0.79 - p/2) * elqparticles[i].best_positions[j] + (0.2 + p/2) * mBestPos[sn][j] + 0.02 * torch::randn_like(elqparticles[i].best_positions[j]);

		auto L = torch::abs(center - elqparticles[i].positions[j]);
                auto direction = torch::where(torch::rand_like(L) < 0.5, 1.0, -1.0);
                /* 
		double u = rand_gen();
		new_pos[j] = center + direction * L * log(1.0 / torch::tensor(u, tsoptions)) * beta * elqparticles[i].search_scale * global_scale;
                */

                auto u = torch::rand_like(center);
                u.clamp_(1.0e-4, 1.0 - 1.0e-4);
                new_pos[j] = center + direction * L * torch::log(1.0 / u) * beta * elqparticles[i].search_scale * global_scale;

                if (elqparticles[i].is_explorer) 
		{
		    if (rand_gen() < 0.5)
		    {	    
                        auto L = torch::abs(gBestPos[sn][j] - elqparticles[i].positions[j]);
                        auto direction = torch::where(torch::rand_like(L) < 0.5, 1.0, -1.0);
                        double lvbeta = 1.0 + 0.9 * progress;
		        double levy_step = rand_gen.levy(lvbeta);
    
                        // 自适应缩放系数，避免 early jump 太大
                        double scale = lvy_scale * elqparticles[i].search_scale;
    
                        // 限幅防发散
                        levy_step = std::clamp(levy_step, -GMAX * 0.8, GMAX * 0.8);

                        new_pos[j] += scale * levy_step * direction * L;
		    }
		    else
		    {
			new_pos[j] += (0.1 + 0.1 * (1.0 - progress)) * torch::randn_like(new_pos[j]);    
		    }
                }

                new_pos[j].clamp_(-GMAX, GMAX);
            }

            std::vector<torch::Tensor> vec;
            for (auto &p : new_pos) vec.push_back(p.to(torch::kFloat));	    
            iPEPS sq(vec);
            auto energy = sq.groundEnergy(i, 'C', chi, {1.0e-4, 1.0e-1}, true, false, false, printable).to(torch::kF64);
            double delta = energy.item<double>() - elqparticles[i].best_energy.item<double>();
            bool accept = false;
            if (delta < 0) 
	    {
                accept = true;
                elqparticles[i].stagnation_count = 0;
            } 
	    else if (sst[sn].currentTp > 0) 
	    {
                if (rand_gen() < exp(-delta / sst[sn].currentTp)) 
		{
                    accept = true;
                    elqparticles[i].stagnation_count++;
                }
            }

            if (accept) 
	    {
                elqparticles[i].best_energy = energy.clone();
                for (int j = 0; j < LY * LX; ++j) elqparticles[i].best_positions[j] = new_pos[j].clone();

                #pragma omp critical
                if (energy.item<double>() < gBest[sn].item<double>()) 
		{
                    gBest[sn] = energy.clone();
                    for (int j = 0; j < LY * LX; ++j) gBestPos[sn][j] = new_pos[j].clone();
                }
            } 
	    else elqparticles[i].stagnation_count++;

            // === 重启机制：当个体停滞过久，注入高斯扰动重启 ===
            if (elqparticles[i].stagnation_count > STAGNATION_THRESHOLD) 
	    {
                for (int j = 0; j < LY * LX; ++j) new_pos[j] += 0.5 * GMAX * torch::randn_like(new_pos[j]);
                elqparticles[i].stagnation_count = 0;
            }

            elqparticles[i].positions = std::move(new_pos);
  	}

        if (iter % 20 == 0) 
	{
	    for (int sn = 0; sn < SUBSWARMS; sn++)
	    {	    
                cout << "iter = " << iter << " gBest = " << gBest[sn].item<double>() / (LX * LY) << std::endl;
                prev_gBest[sn] = gBest[sn].item<double>();
	    }
        }

        if (iter > WarmNum && iter < WarmNum + ELQ_ITERNUM * 0.9 && iter%20 == 0)  // exchange temperature
	{
	    bool ceo = rand_gen() > 0.1;
            
            for(int ct = 0; ct < 2; ct++)
	    {	    
		std::vector<int> subswarm_indices(SUBSWARMS);
                std::iota(subswarm_indices.begin(), subswarm_indices.end(), 0);
                std::sort(subswarm_indices.begin(), subswarm_indices.end(), [&](int a, int b)
                {
                    return sst[a].currentTp < sst[b].currentTp;
                });

		if (ct == 0)
		{
	            for (int ss : subswarm_indices) std::cout << "ss = " << ss <<"  " << sst[ss].currentTp << " " <<gBest[ss].item<double>()/LX/LY << std::endl;
	            std::cout << std::endl;
                }

	        for (int i = SUBSWARMS-1; i > 0; i--)
	        {
		    int ss = subswarm_indices[i];   
	            int ssn = subswarm_indices[i-1];

	            double delta = (1.0/sst[ss].currentTp-1.0/sst[ssn].currentTp)*(gBest[ssn].item<double>()- gBest[ss].item<double>()); 	
                    double xi = exp(-delta);
		    if ((ceo && xi > 1.0) || (!ceo && (xi > 1.0 || xi > rand_gen())))
		    {
		        std::swap(sst[ss].initialTp, sst[ssn].initialTp);
	                std::swap(sst[ss].currentTp, sst[ssn].currentTp);
	                std::swap(sst[ss].finalTp, sst[ssn].finalTp);	  
		        std::swap(sst[ss].levy_scale, sst[ssn].levy_scale);

		        std::swap(subswarm_indices[i], subswarm_indices[i-1]); 
		    }
	        }
	    }

	    std::vector<int> subswarm_indices(SUBSWARMS);
	    std::iota(subswarm_indices.begin(), subswarm_indices.end(), 0);
            std::sort(subswarm_indices.begin(), subswarm_indices.end(), [&](int a, int b)
            {
                return sst[a].currentTp < sst[b].currentTp;
            });

            for (int ss : subswarm_indices) std::cout << "ss = " << ss <<"  " << sst[ss].currentTp << " " <<gBest[ss].item<double>()/LX/LY << std::endl;
        }

	if (iter >= WarmNum && iter < WarmNum + ELQ_ITERNUM * 0.7 && iter%41 == 0)  // exchanging particles between nearest temperature
	{
	    std::vector<int> subswarm_indices(SUBSWARMS);
            std::iota(subswarm_indices.begin(), subswarm_indices.end(), 0);
            std::sort(subswarm_indices.begin(), subswarm_indices.end(), [&](int a, int b)
            {
                return sst[a].currentTp < sst[b].currentTp;
            });

            for (int ig = 0; ig < SUBSWARMS-1; ig++)
	    {
		int sn = subswarm_indices[ig];    
		int nsn = subswarm_indices[ig+1];   
	       
		for (int ip = 0; ip < GPNUM; ++ip)
                {
                    int gidx = sn * GPNUM + ip;
                    int i = mapSS2P[gidx];
                    energy_with_index[sn][ip] = { elqparticles[i].best_energy.item<double>(), i };
                }

		std::sort(energy_with_index[sn].begin(), energy_with_index[sn].end());

		for (int ip = 0; ip < GPNUM; ++ip)
                {
                    int gidx = nsn * GPNUM + ip;
                    int i = mapSS2P[gidx];
                    energy_with_index[nsn][ip] = { elqparticles[i].best_energy.item<double>(), i };
                }

                std::sort(energy_with_index[nsn].begin(), energy_with_index[nsn].end());

		int exip = 0;
		double r = rand_gen();
		while (r > exweight[exip]) exip++;
		std::cout <<"exip = " << exip << std::endl;
		assert(exip < GPNUM);

		{
                    int mnl = energy_with_index[sn][exip].second;
		    int mnr = energy_with_index[nsn][exip].second;

	            std::swap(elqparticles[mnl].gidx, elqparticles[mnr].gidx);	
		    mapSS2P[elqparticles[mnl].gidx] = mnl;
		    mapSS2P[elqparticles[mnr].gidx] = mnr;   

		    assert(elqparticles[mnl].gidx/GPNUM == nsn);
		    assert(elqparticles[mnr].gidx/GPNUM == sn);
		}
	    } 
	}

	if (iter == WarmNum)
	{
            #pragma omp parallel for schedule(dynamic)		
	    for (int sn = 0; sn < SUBSWARMS; sn++)	
	    {
		//#pragma omp parallel num_threads(4)    
		{    
		    int chi = KAI;    
//		    localLBFGSOptimization(200+sn, gBestPos[sn], gBest[sn], chi, {1.0e-3, 1.0e-1}, 10, 3, true);    
                    localAdamOptimization(200+sn, gBestPos[sn], gBest[sn], chi, {1.0e-3, 1.0e-1}, 5, true);
		}
	    }
	}
	else if (iter > WarmNum && (iter - WarmNum)%30 == 0)  // local optimization
	{
	    std::vector<int> subswarm_indices(SUBSWARMS);
            std::iota(subswarm_indices.begin(), subswarm_indices.end(), 0);
            std::sort(subswarm_indices.begin(), subswarm_indices.end(), [&](int a, int b)
            {
                return gBest[a].item<double>() < gBest[b].item<double>();
            });

            for (int sn = 0; sn < SUBSWARMS; sn++) sst[sn].explorer_count = GPNUM;
            if (progress > 0.5) sst[subswarm_indices[0]].explorer_count = std::min(std::max((int)(GPNUM * EXPLORER_RATIO), GPNUM - (int)(GPNUM * progress * progress)), GPNUM-1);

            //auto last_start = subswarm_indices.begin() + 1;
            //auto last_end = subswarm_indices.end();
	
	    bool isaccepted[SUBSWARMS];	
	    #pragma omp parallel for schedule(dynamic)       
            for (int sn = 0; sn < SUBSWARMS; sn++)
            {	
	        energy_with_index[sn].resize(GPNUM);
                for (int ip = 0; ip < GPNUM; ++ip)
                {
                    int gidx = sn * GPNUM + ip;
                    int i = mapSS2P[gidx];
                    energy_with_index[sn][ip] = { elqparticles[i].best_energy.item<double>(), i };
                }

                std::sort(energy_with_index[sn].begin(), energy_with_index[sn].end());	

		for (int igp = 0; igp < GPNUM; igp++)
                {
                    int i = energy_with_index[sn][igp].second;
                    if (igp >= GPNUM - sst[sn].explorer_count) elqparticles[i].is_explorer = true;
                    else elqparticles[i].is_explorer = false;
                }

                int best_idx = energy_with_index[sn][0].second;
                int sdbest_idx = energy_with_index[sn][1].second;

		bool islowered = true;
		int prefidx = 200;
		int lln = 2;
                if (progress > 0.7) 
		{
		    islowered = false;
		    prefidx = 300;
		    lln = 2;
                }

		//#pragma omp parallel num_threads(4)
		{
		    int chi = KAI;

		    isaccepted[sn] = localLBFGSOptimization(prefidx+sn, elqparticles[best_idx].best_positions, gBestPos[sn], gBest[sn], chi, {1.0e-4, 1.0e-2}, maxLBFGSiter, lln, islowered);
  		}

                if (isaccepted[sn]) rejectedfreq[sn] = 0;
                else rejectedfreq[sn]++;

		int worst_idx = energy_with_index[sn].back().second;
                int second_worst_idx = energy_with_index[sn][GPNUM - 2].second;

                auto perturb_particle = [&](int idx, double sigma, bool is_large)
                {
                    for (int j = 0; j < LY*LX; j++)
                    {
                        if (is_large)
                        {
                            auto u = torch::randn_like(elqparticles[idx].positions[j]) * sigma * GMAX;
                            auto v = torch::randn_like(elqparticles[idx].positions[j]);
                            elqparticles[idx].positions[j] += u / torch::pow(torch::abs(v) + 1.0e-8, 1.0 / 1.5);
                        }
                        else
                        {
                            auto noise = torch::randn_like(elqparticles[idx].positions[j]) * sigma * GMAX;
                            elqparticles[idx].positions[j] += noise;
                        }

                        elqparticles[idx].positions[j].clamp_(-GMAX, GMAX);
                    }
                    elqparticles[idx].stagnation_count = 0;
                };

                if (isaccepted[sn])
                {
                    double sigma_small = 0.02 * (1.0 - progress);
                    perturb_particle(worst_idx, sigma_small, false);
                    perturb_particle(second_worst_idx, sigma_small*0.1, false);
                }
                else if (rejectedfreq[sn] >= 2)
                {
                    int ip = (int)(rand_gen()*GPNUM)%(GPNUM-1) + 1;
                    int idx = energy_with_index[sn][ip].second;

                    isaccepted[sn] = localLBFGSOptimization(prefidx+sn, elqparticles[idx].best_positions, gBestPos[sn], gBest[sn], chi, {1.0e-4, 1.0e-2}, maxLBFGSiter, lln, islowered);

                    if (isaccepted[sn]) rejectedfreq[sn] = 0;
                    else rejectedfreq[sn]++;
 
		    if (rejectedfreq[sn] >= 4 && sst[sn].explorer_count == GPNUM)
		    {
			chaoticPerturbateGBest(gBestPos[sn], chaos[sn], rejectedfreq[sn], progress, GMAX);

			iPEPS sq(gBestPos[sn]);
			gBest[sn] = sq.groundEnergy(10000, 'I', chi, {1.0e-4, 1.0e-2}, true);
                        //gBest[sn] = torch::tensor({INITIAL_BEST_ENERGY * (1.0 + 0.1 * (1.0 - progress))}, torch::kF64);

                        double sigma_large = 0.1 * (1.0 - progress * 0.8);
                        for (int ip = 0; ip < GPNUM; ip++)
                        {
                            int idx = energy_with_index[sn][ip].second;
                            if (rand_gen() < 0.6) 
			    {
				perturb_particle(idx, sigma_large, true);
                                sigma_large *= 1.2;
			    }
			    else
			    {
				chaoticPerturbateGBest(elqparticles[idx].positions, chaos[sn], rejectedfreq[sn], progress, GMAX);    
			    }

                            if (ip >= GPNUM/2) elqparticles[idx].best_energy = torch::tensor({INITIAL_BEST_ENERGY}, torch::kF64);
			}

			sst[sn].currentTp *= 8;
			sst[sn].levy_scale *= 4;
                    }
                }
            }

	    if (maxLBFGSiter < 15) maxLBFGSiter++;

	    std::cout <<"Acceptance ratio is : " << std::endl;
            for (int sn = 0; sn < SUBSWARMS; sn++) std::cout << isaccepted[sn] <<"  ";
            std::cout << std::endl;
            for (int sn = 0; sn < SUBSWARMS; sn++) std::cout << rejectedfreq[sn] <<"  ";
            std::cout << std::endl;
            for (int sn = 0; sn < SUBSWARMS; sn++) std::cout << gBest[sn].item<double>()/LX/LY << " ";
            std::cout << std::endl;
	}

        auto timer_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start);

        std::cout << "Iteration "<<iter <<" completed in " << duration.count()/1000.0 << " S\n" << std::endl;
    }


    for (int sn = 0; sn < SUBSWARMS; sn++)
    {
        for (int y = 0; y < LY; y++)
        {
            for (int x = 0; x < LX; x++)
            {
                stringname(64, filename, sn, "Gyxx", y, x);
                torch::save(gBestPos[sn][y*LX+x], filename);
            }
        }
    }

    std::cout <<"PSO finished! Start auto diff optimization." << std::endl;

    #pragma omp parallel for schedule(dynamic)		
    for (int sn = 0; sn < SUBSWARMS; sn++)	
    {
	//#pragma omp parallel num_threads(4)    
	{    
            int chi = KAI;    
	    localLBFGSOptimization(400+sn, gBestPos[sn], gBest[sn], chi, {1.0e-4, 1.0e-3}, 15, 20);
	}

	std::cout <<" sn = " << sn <<"   E0 = "<< gBest[sn].item<double>()/LY/LX << std::endl; 
    }

    /*final calculations*/

    std::cout <<"========Optimization finished, now enter the final simulations!==========" << std::endl;


//    #pragma omp parallel for num_threads(SUBSWARMS)             
    for (int sn = 0; sn < SUBSWARMS; sn++)
    {
	std::cout <<"***********sn = " << sn << std::endl;    
        //#pragma omp parallel num_threads(4)    
        {
            int chi = KAI;
	    iPEPS sq(gBestPos[sn]);
	    gBest[sn] = sq.groundEnergy(400+sn, 'C', chi, {1.0e-4, 1.0e-3}, true, true, true, true);
        }
	std::cout <<" sn = " << sn <<"   E0 = "<< gBest[sn].item<double>()/LY/LX << std::endl<<std::endl;
	std::cout<<std::endl;
    }

    auto timer_end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(timer_end - timer_start);
    cout << "Multi-scale ELQPSO-SA completed in " << duration.count() / 1000.0 << " seconds" << endl;

    coutfile.flush();
    coutfile.close();
    cout.rdbuf(coutbuff);
    return 0;
}
