//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iostream>
#include <ctime>

using namespace sycl;

// global settings
const int N = 1*1024*1;
const int blocks = 32;

// gpu
class IntelGPUSelector : public device_selector {
 public:
  int operator()(const device& Device) const override {
    const std::string DeviceName = Device.get_info<info::device::name>();
    const std::string DeviceVendor = Device.get_info<info::device::vendor>();

    return Device.is_gpu() && (DeviceName.find("Intel") != std::string::npos) ? 100 : 0;
  }
};

int Serial_Euler_Sieve(queue &q, int *primes, bool *vis){
    clock_t start, end;
    
    for(int i = 2;i<N;++i){
        vis[i] = false;
    }
    int cur = 0;
    
    start = clock();
    for(int i = 2;i<N;++i){
        if(!vis[i]){
            primes[cur++] = i;
        }
        for(int j = 0;j<cur && i*primes[j] < N;++j){
            vis[i*primes[j]] = true;
            if(i % primes[j] == 0) break;
        }
    }
    end = clock();

    return end - start;
}

int Parallel_Euler_Sieve(queue &q, int *primes, bool *vis){
    // record running time
    clock_t start, end;
    
    // init
    for (int i = 2; i < N; ++i) {
        vis[i] = false;
    }
    int cur = 0, cur_prime = 2;
    
    // start
    start = clock();
    while (cur_prime < N){
        if(!vis[cur_prime]){
            primes[cur++] = cur_prime;
        }
        int piece = N / blocks;
        q.parallel_for(range<1>(blocks), [=](id<1> i) {
            int start = (i*piece);
            int end = min(cur,start + piece);
            for (int j = start; j < end && cur_prime*primes[j]<N; ++j) {
                vis[cur_prime * primes[j]] = true;
                if(cur_prime % primes[j] == 0) break;
            }
        }).wait();
        ++cur_prime;
    }
    end = clock();
    
    // running time
    return end - start;
}

// verify
void test(int *primes){
    int i = 0;
    for (; i < N; ++i) {
        if(primes[i] == 0) break;
        std::cout << primes[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "\nparallel total: "<< i << std::endl;
}

int main() {
    
    IntelGPUSelector d;
    queue q(d);
    
    int *primes = malloc_shared<int>(N, q);
    bool *vis = malloc_shared<bool>(N, q);
    
    // Parallel
    std::cout << "used time: " << Parallel_Euler_Sieve(q,primes,vis) << std::endl;
    test(primes);
    
    std::cout << std::endl;
    
    // Serial
    std::cout << "used time: " << Serial_Euler_Sieve(q,primes,vis) << std::endl;
    test(primes);
    
    free(primes, q);
	free(vis, q);
    return 0;
}
