#ifndef _OPTIMIZATION_LOCAL_H
#define _OPTIMIZATION_LOCAL_H

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


void localLBFGSOptimization(const int &fidx, std::vector<torch::Tensor>& solution, torch::Tensor &energy, const int &chi, const std::pair<double, double> &epspr, const int &maxlbfgsiter=10, const int&lln = 1, const bool &islowered = false) 
{
    try 
    {
        char filename[120];
        stringname(80, filename, "Energy_Cluster", fidx);

        ofstream file;
        file.open(filename, ios::app | ios::out);
        if (!file) return;

        file.precision(10);
        file.width(15);
        //   file.fill('0');
        file.setf(ios::left);


        // 1. 创建参数的深拷贝并确保它们是叶子节点
        std::vector<torch::Tensor> params_to_optimize;
        if (islowered) 
	{
	    for (auto& t : solution) params_to_optimize.push_back(t.to(torch::kFloat).clone().detach().requires_grad_(true));
	}
	else 
	{
	    for (auto& t : solution) params_to_optimize.push_back(t.clone().detach().requires_grad_(true));
        }

        // 2. 创建iPEPS实例
        iPEPS sq(params_to_optimize, true);
        
        // 3. 动态调整学习率
        double current_energy = energy.item<double>();
        double fslr = 0.2 * (1.0 + 0.5 * exp(-current_energy / 10.0));
        
        torch::optim::LBFGS optimizer(sq.parameters(), torch::optim::LBFGSOptions().lr(fslr).max_iter(maxlbfgsiter).line_search_fn("strong_wolfe"));
        
        int ecount = 0;
        auto closure = [&]() -> torch::Tensor 
	{
            optimizer.zero_grad();
            auto local_energy = sq.groundEnergy(fidx, 'C', chi, epspr, true, false, true, true);
            local_energy.backward();
            
            // 梯度裁剪防止数值不稳定
            //torch::nn::utils::clip_grad_norm_(params, 1.0);
       
            file << ecount << "     "<< local_energy.item<double>() <<"    "<<  (local_energy/(LX*LY)).item<double>() << "  LBFGS" <<std::endl;	    
            ecount++;
            return local_energy;
        };

        // 6. 执行优化步骤
        for (int ln = 0; ln < lln; ln++) optimizer.step(closure);

	optimizer.state().clear();
        optimizer.zero_grad();

	file.close();

        // 7. 更新结果
        auto optimized_energy = sq.groundEnergy(fidx, 'C', chi, epspr, true);
        energy = optimized_energy.to(torch::kF64).clone();

        // 将优化后的参数复制回solution，但不保留梯度信息
        solution.clear();
        if (islowered) 
	{
	    for (auto& t : params_to_optimize) solution.push_back(t.to(torch::kF64).clone().detach());
	}
	else 
	{
	    for (auto& t : params_to_optimize) solution.push_back(t.clone().detach());
        }

	std::vector<torch::Tensor>().swap(params_to_optimize);

        cout << "[Local LBFGS] Energy improved from " << current_energy / (LX * LY) 
             << " to " << energy.item<double>() / (LX * LY) 
             << " | Steps: " << ecount << endl;
    } 
    catch (const c10::Error& e) 
    {
        cerr << "[C10 Exception in LBFGS]: " << e.what() << endl;
        // 恢复原始solution以防出错
        for (size_t i = 0; i < solution.size(); ++i) solution[i] = solution[i].clone().detach();
        throw;
    }
    catch (const std::exception& e) 
    {
        cerr << "[STD Exception in LBFGS]: " << e.what() << endl;
        // 恢复原始solution以防出错
        for (size_t i = 0; i < solution.size(); ++i) solution[i] = solution[i].clone().detach();
        throw;
    }
}

bool localLBFGSOptimization(const int &fidx, const std::vector<torch::Tensor>& solution_in, std::vector<torch::Tensor> &solution_out, torch::Tensor &energy_io, const int &chi, const std::pair<double, double> &epspr, const int &maxlbfgsiter=10, const int&lln = 1, const bool &islowered = false) 
{
    try 
    {
        // 1. 创建参数的深拷贝并确保它们是叶子节点
        std::vector<torch::Tensor> wavefunc;
        if (islowered)
        {
            for (auto& t : solution_in) wavefunc.push_back(t.to(torch::kFloat).clone().detach().requires_grad_(true));
        }
        else
        {
            for (auto& t : solution_in) wavefunc.push_back(t.clone().detach().requires_grad_(true));
        }


        iPEPS sq(wavefunc, true);
        
        double current_energy = energy_io.item<double>();
        double fslr = 0.2 * (1.0 + 0.5 * exp(-current_energy / 10.0));
        
        torch::optim::LBFGS optimizer(sq.parameters(), torch::optim::LBFGSOptions().lr(fslr).max_iter(maxlbfgsiter).line_search_fn("strong_wolfe"));
        
        // 5. 定义闭包函数
        int ecount = 0;
	torch::Tensor closure_energy;
        auto closure = [&]() -> torch::Tensor 
	{
            optimizer.zero_grad();
            auto local_energy = sq.groundEnergy(fidx, 'C', chi, epspr, true, false, true, true);
	    closure_energy = local_energy;
            local_energy.backward();
            
            // 梯度裁剪防止数值不稳定
            //torch::nn::utils::clip_grad_norm_(params, 1.0);
            
            ecount++;
            return local_energy;
        };

        // 6. 执行优化步骤
        for (int ln = 0; ln < lln; ln++) optimizer.step(closure);

	optimizer.state().clear();

	if (closure_energy.to(torch::kF64).item<double>() < energy_io.to(torch::kF64).item<double>())
	{
	    energy_io = closure_energy.to(torch::kF64).clone().detach();

        // 将优化后的参数复制回solution，但不保留梯度信息
            solution_out.clear();
            if (islowered) 
	    {
	        for (auto& t : wavefunc) solution_out.push_back(t.to(torch::kF64).clone().detach());
	    }
	    else 
	    {
	        for (auto& t : wavefunc) solution_out.push_back(t.clone().detach());
            }

            cout << "[Local LBFGS] Energy improved from " << current_energy / (LX * LY) 
                 << " to " << energy_io.item<double>() / (LX * LY) 
                 << " | Steps: " << ecount << endl;

	    std::vector<torch::Tensor>().swap(wavefunc);

	    return true;
	}
	else 
	{
	    return false;
	}
    } 
    catch (const c10::Error& e) 
    {
        cerr << "[C10 Exception in LBFGS]: " << e.what() << endl;
        throw;
    }
    catch (const std::exception& e) 
    {
        cerr << "[STD Exception in LBFGS]: " << e.what() << endl;
        throw;
    }

    return false;
}


void localAdamOptimization(const int &fidx, std::vector<torch::Tensor>& solution, torch::Tensor &energy, const int &chi, const std::pair<double, double> &epspr, const int &maxadamsteps=20, const bool &islowered = false) 
{
    try 
    {
        // 1. 创建参数的深拷贝并确保它们是叶子节点
        std::vector<torch::Tensor> params_to_optimize;
        if (islowered) 
	{
	    for (auto& t : solution) params_to_optimize.push_back(t.to(torch::kFloat).clone().detach().requires_grad_(true));
	}
	else 
	{
	    for (auto& t : solution) params_to_optimize.push_back(t.clone().detach().requires_grad_(true));
        }

        // 2. 创建iPEPS实例
        iPEPS sq(params_to_optimize, true);
        
        // 3. 动态调整学习率
        double current_energy = energy.item<double>();
        double fslr = 0.3 * (1.0 + 0.5 * exp(-current_energy / 10.0));
        
        auto opts = torch::optim::AdamOptions().lr(fslr);
        torch::optim::Adam optimizer(sq.parameters(), opts);

        torch::Tensor closure_energy;
        for (int i = 0; i < maxadamsteps; i++)
        {
            optimizer.zero_grad();
            auto local_energy = sq.groundEnergy(fidx, 'C', chi, epspr, true, false, true, true);
            closure_energy = local_energy;
            local_energy.backward();

            optimizer.step();
        }

	optimizer.state().clear();
        optimizer.zero_grad();

        // 7. 更新结果
        auto optimized_energy = sq.groundEnergy(fidx, 'C', chi, epspr, true);
        energy = optimized_energy.to(torch::kF64).clone();

        // 将优化后的参数复制回solution，但不保留梯度信息
        solution.clear();
        if (islowered) 
	{
	    for (auto& t : params_to_optimize) solution.push_back(t.to(torch::kF64).clone().detach());
	}
	else 
	{
	    for (auto& t : params_to_optimize) solution.push_back(t.clone().detach());
        }

	std::vector<torch::Tensor>().swap(params_to_optimize);

        cout << "[Local Adam] Energy improved from " << current_energy / (LX * LY) 
             << " to " << energy.item<double>() / (LX * LY) << endl;
    } 
    catch (const c10::Error& e) 
    {
        cerr << "[C10 Exception in Adam]: " << e.what() << endl;
        // 恢复原始solution以防出错
        for (size_t i = 0; i < solution.size(); ++i) solution[i] = solution[i].clone().detach();
        throw;
    }
    catch (const std::exception& e) 
    {
        cerr << "[STD Exception in Adam]: " << e.what() << endl;
        // 恢复原始solution以防出错
        for (size_t i = 0; i < solution.size(); ++i) solution[i] = solution[i].clone().detach();
        throw;
    }
}

bool localAdamOptimization(const int &fidx, const std::vector<torch::Tensor>& solution_in, std::vector<torch::Tensor> &solution_out, torch::Tensor &energy_io, const int &chi, const std::pair<double, double> &epspr, const int &maxadamsteps = 20, const bool &islowered = false) 
{
    try 
    {
        // 1. 创建参数的深拷贝并确保它们是叶子节点
        std::vector<torch::Tensor> wavefunc;
        if (islowered)
        {
            for (auto& t : solution_in) wavefunc.push_back(t.to(torch::kFloat).clone().detach().requires_grad_(true));
        }
        else
        {
            for (auto& t : solution_in) wavefunc.push_back(t.clone().detach().requires_grad_(true));
        }


        // 2. 创建iPEPS实例
        iPEPS sq(wavefunc, true);
        
        // 3. 动态调整学习率
        double current_energy = energy_io.item<double>();
        double fslr = 0.3 * (1.0 + 0.5 * exp(-current_energy / 10.0));
       
        auto opts = torch::optim::AdamOptions().lr(fslr);
        torch::optim::Adam optimizer(sq.parameters(), opts);

        int ecount = 0;
        torch::Tensor closure_energy;
	for (int i = 0; i < maxadamsteps; i++)
        {
            optimizer.zero_grad();
            auto local_energy = sq.groundEnergy(fidx, 'C', chi, epspr, true, false, true, true);
            closure_energy = local_energy;
            local_energy.backward();

            ecount++;

            optimizer.step();
        }

	optimizer.state().clear();

	if (closure_energy.to(torch::kF64).item<double>() < energy_io.to(torch::kF64).item<double>())
	{
	    energy_io = closure_energy.to(torch::kF64).clone().detach();

        // 将优化后的参数复制回solution，但不保留梯度信息
            solution_out.clear();
            if (islowered) 
	    {
	        for (auto& t : wavefunc) solution_out.push_back(t.to(torch::kF64).clone().detach());
	    }
	    else 
	    {
	        for (auto& t : wavefunc) solution_out.push_back(t.clone().detach());
            }

            cout << "[Local Adam] Energy improved from " << current_energy / (LX * LY) 
                 << " to " << energy_io.item<double>() / (LX * LY) 
                 << " | Steps: " << ecount << endl;

	    std::vector<torch::Tensor>().swap(wavefunc);

	    return true;
	}
	else 
	{
	    return false;
	}
    } 
    catch (const c10::Error& e) 
    {
        cerr << "[C10 Exception in Adam]: " << e.what() << endl;
        throw;
    }
    catch (const std::exception& e) 
    {
        cerr << "[STD Exception in Adam]: " << e.what() << endl;
        throw;
    }

    return false;
}

bool localAdamLBFGSOptimization(const int &fidx, const std::vector<torch::Tensor>& solution_in, std::vector<torch::Tensor> &solution_out, torch::Tensor &energy_io, const int &chi, const std::pair<double, double> &epspr, const int &maxadamsteps = 20, const int &maxlbfgsiter=10, const int&lln = 1, const bool &islowered = false) 
{
    try 
    {
        // 1. 创建参数的深拷贝并确保它们是叶子节点
        std::vector<torch::Tensor> wavefunc;
        if (islowered)
        {
            for (auto& t : solution_in) wavefunc.push_back(t.to(torch::kFloat).clone().detach().requires_grad_(true));
        }
        else
        {
            for (auto& t : solution_in) wavefunc.push_back(t.clone().detach().requires_grad_(true));
        }


        iPEPS sq(wavefunc, true);
        
        double current_energy = energy_io.item<double>();
        double fslr = 0.3 * (1.0 + 0.5 * exp(-current_energy / 10.0));
       
        auto opts = torch::optim::AdamOptions().lr(fslr);
        torch::optim::Adam optimizerAdam(sq.parameters(), opts);

        int ecount = 0;
        torch::Tensor closure_energy;
	for (int i = 0; i < maxadamsteps/2; i++)
        {
            optimizerAdam.zero_grad();
            auto local_energy = sq.groundEnergy(fidx, 'C', chi, epspr, true, false, true, true);
            closure_energy = local_energy;
            local_energy.backward();

            ecount++;

            optimizerAdam.step();
        }

	optimizerAdam.state().clear();

        fslr = 0.1 * (1.0 + 0.5 * exp(-current_energy / 10.0));

        torch::optim::LBFGS optimizerLBFGS(sq.parameters(), torch::optim::LBFGSOptions().lr(fslr).max_iter(maxadamsteps/2).line_search_fn("strong_wolfe"));

        auto closure = [&]() -> torch::Tensor
        {
            optimizerLBFGS.zero_grad();
            auto local_energy = sq.groundEnergy(fidx, 'C', chi, epspr, true, false, true, true);
            closure_energy = local_energy;
            local_energy.backward();

            // 梯度裁剪防止数值不稳定
            //torch::nn::utils::clip_grad_norm_(params, 1.0);

            ecount++;
            return local_energy;
        };

        optimizerLBFGS.step(closure);

        optimizerLBFGS.state().clear();

	if (closure_energy.to(torch::kF64).item<double>() < energy_io.to(torch::kF64).item<double>())
	{
	    energy_io = closure_energy.to(torch::kF64).clone().detach();

        // 将优化后的参数复制回solution，但不保留梯度信息
            solution_out.clear();
            if (islowered) 
	    {
	        for (auto& t : wavefunc) solution_out.push_back(t.to(torch::kF64).clone().detach());
	    }
	    else 
	    {
	        for (auto& t : wavefunc) solution_out.push_back(t.clone().detach());
            }

            cout << "[Local Adam] Energy improved from " << current_energy / (LX * LY) 
                 << " to " << energy_io.item<double>() / (LX * LY) 
                 << " | Steps: " << ecount << endl;

	    std::vector<torch::Tensor>().swap(wavefunc);

	    return true;
	}
	else 
	{
	    return false;
	}
    } 
    catch (const c10::Error& e) 
    {
        cerr << "[C10 Exception in LBFGS]: " << e.what() << endl;
        throw;
    }
    catch (const std::exception& e) 
    {
        cerr << "[STD Exception in LBFGS]: " << e.what() << endl;
        throw;
    }

    return false;
}

#endif
