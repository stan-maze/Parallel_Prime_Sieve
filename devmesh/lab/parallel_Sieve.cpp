#include <CL/sycl.hpp>
#include <iostream>
#include<ctime>

using namespace sycl;

int count(int *primes, const int end){
    int cnt = 0;
    for (int i = 2; i < end; i++) {
        if (primes[i]){
            cnt++;
            //std::cout << i << " ";
        }
    }
    return cnt;
}

int parallel_sieve(queue &q, int *primes, const int N, const int blocks){
    clock_t start, end;
    for (int i = 2; i < N; i++) {
        primes[i] = 1;
    }
    int curr_prime = 2;
    start = clock();
    while (primes[curr_prime]){
        int bounds = N / curr_prime + 1;
        int piece = (bounds/blocks + 1)*curr_prime;
        q.parallel_for(range<1>(blocks), [=](id<1> i) {
            int start = curr_prime + i*piece;
            int end = i<(blocks-1)?(start+piece):N;
            // int end = start + piece;
            // end = end > N ? N : end;
            //if(i == blocks - 1){
            //   end = N;
            //}else{
            //   end = start + piece;
            //}
            for (int j = start; j < end; j += curr_prime) {
                primes[j] = 0;
            }
        }).wait();
        primes[curr_prime ++] = 1;
        while (curr_prime <= sqrt(N) && !primes[curr_prime]) {
            curr_prime ++;
        }
    }
    end = clock();
    return end-start;
}

int serial_sieve(int *primes, const int N){
    clock_t start, end;
    for (int i = 2; i < N; i++) {
        primes[i] = 1;
    }
    start = clock();
    for (int i = 2; i < sqrt(N); ++i) {
        if (primes[i]) {
            for (int j = 2 * i; j < N; j += i) {
                primes[j] = 0;
            }
        }
    }
    end = clock();
    return end-start;
}

void test(queue q, const int N, const int blocks){
    //queue q;
    int *primes = malloc_shared<int>(N, q);
    std::cout<<"-------------------------------------\n";
    std::cout<<"Num of Numbers: "<<N<<"\tNum of blocks:"<<blocks<<std::endl;
    std::cout<< "parallel used time: "<< parallel_sieve(q, primes, N, blocks) <<std::flush;;
    //std::cout<<"\ttotal: "<< count(primes, N);
    std::cout<< "\nserial used time: "<< serial_sieve(primes, N) <<std::flush;
    //std::cout<<"\t total: "<< count(primes, N);
    std::cout<<"\n-------------------------------------"<<std::endl;;
    free(primes, q);
}
int main() {
    queue q(gpu_selector_v);
    int Ns[] = {1024, 1024*1024, 1024*1024*32, 1024*1024*128, 1024*1024*512, 1024*1024*1024};
    int Nblocks[] = {8, 16, 32, 64, 128, 256, 512};
    // 由于的devcloud只允许程序运行一分钟时间, 所以网格要手动调整多次
    for(int i=0; i<2; i++){
        for(int j=0; j<7; j++){
            test(q, Ns[i], Nblocks[j]);
        }
    }
    return 0;
}
