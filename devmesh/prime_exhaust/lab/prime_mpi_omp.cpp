#include "mpi.h" 
#include <stdio.h> 
#include <math.h> 
#include <sys/time.h>
#include <omp.h>
#include <iostream>
using namespace std;

//数据量
constexpr long TotalNumStep = 1024;

void exhaust(int* flag_per_rank, int myid, int numprocs, int num_step){
    char machine_name[MPI_MAX_PROCESSOR_NAME]; 
    int namelen;
    int is_cpu=true;
    int* flag_limit = new int[num_step];
    for(int i=1;i<num_step;i++)flag_limit[i]=sqrt(i*numprocs+myid);
    MPI_Get_processor_name(machine_name,&namelen);
    
     #pragma omp target map(from:is_cpu) map(tofrom:flag_per_rank[0:num_step])
    {  
        
        #pragma omp teams distribute parallel for simd
        for (int k=0; k< num_step; k++) {
            if (k==0) is_cpu=omp_is_initial_device();
            //int limit (k*numprocs+myid);
            for(int i=2;i<(k*numprocs+myid);i++){
                if((k*numprocs+myid)%i==0){
                    flag_per_rank[k]=false;
                    break;
                }
            }
        }
    }
    cout << "Rank " << myid << " of " << numprocs
         << " runs on: " << machine_name
         << ", uses device: " << (is_cpu?"CPU":"GPU")
         << "\n";
}

int main(int argc, char* argv[])
{
    int myid, numprocs; 
    struct timeval ts,te;
    MPI_Status status;
    int* flag = new int[TotalNumStep];
    for(int i=0;i<TotalNumStep;i++)flag[i]=true;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid); 
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs); 
    gettimeofday(&ts,NULL);
    int num_step = TotalNumStep / numprocs;
    int* flag_per_rank = new int[num_step];
    for(int i=0;i<num_step;i++)flag_per_rank[i] = true;
    exhaust(flag_per_rank, myid, numprocs, num_step);
    
    if(myid==0)
    {
        int** all_flag_per_rank = new int*[numprocs];
        all_flag_per_rank[0] = flag_per_rank;
        for(int i=1;i<numprocs;i++){
            all_flag_per_rank[i] = new int[num_step];
            for(int j=0;j<num_step;j++)all_flag_per_rank[i][j] = true;
        }
        /*int* flag_per_rank_1 = new int[num_step];
        for(int i=0;i<num_step;i++)flag_per_rank_1[i] = true;
        int* flag_per_rank_2 = new int[num_step];
        for(int i=0;i<num_step;i++)flag_per_rank_2[i] = true;
        int* flag_per_rank_3 = new int[num_step];
        for(int i=0;i<num_step;i++)flag_per_rank_3[i] = true;*/
        
        for(int i=1;i<numprocs;i++){
            MPI_Recv(all_flag_per_rank[i],num_step,MPI_INT,i,99,MPI_COMM_WORLD, &status);
        }
        /*MPI_Recv(flag_per_rank_1,num_step,MPI_INT,1,99,MPI_COMM_WORLD, &status);
        MPI_Recv(flag_per_rank_2,num_step,MPI_INT,2,99,MPI_COMM_WORLD, &status);
        MPI_Recv(flag_per_rank_3,num_step,MPI_INT,3,99,MPI_COMM_WORLD, &status);*/

        /*for(int i=0;i<num_step;i++){
            flag[i*numprocs+0]=flag_per_rank[i];
            flag[i*numprocs+1]=flag_per_rank_1[i];
            flag[i*numprocs+2]=flag_per_rank_2[i];
            flag[i*numprocs+3]=flag_per_rank_3[i];
        }*/
        for(int i=0;i<numprocs;i++){
            for(int j=0;j<num_step;j++){
                flag[j*numprocs+i]=all_flag_per_rank[i][j];
            }
        }
        gettimeofday(&te,NULL);
        std::cout<<"total time: "<<((te.tv_sec-ts.tv_sec)*1000+(te.tv_usec-ts.tv_usec)/1000)<<"ms"<<endl;
        for(int i=2;i<TotalNumStep;i++)if(flag[i])cout<<i<<" ";
        cout<<endl;
    }
    else{
        MPI_Send(flag_per_rank,num_step,MPI_INT,0,99,MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
