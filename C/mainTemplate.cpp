#include <mpi.h>

//Part 1: Implemnt the Basic Blocking Method (From HW 1)
void blocked_matmat (int n, double* A, double* B, double* C, int n_iter){

}


//Part 2: Implement the Mat-Mat Algo's (From HW 2)
void fox_matmat (double* A, double* B, double* C, int n, int sq_num_procs, int rank_row, int rank_col){


}

void cannon_matmat (double* A, double* B,double* C,int n, int sq_num_procs, int rank_row, int rank_col){

}

//Part 3: Implement 2 of the following with OneSided Communication (Make sure other returns -1)
void rma_blocked (int n, double* A, double* B, double* C, int n_iter) {
	return NULL;
}

void rma_fox (double* A, double* B, double* C, int n, int sq_num_procs, int rank_row, int rank_col) {
	return NULL;
}

void rma_cannon(double* A, double* B, double* C,int n, int sq_num_procs, int rank_row, int rank_col){
	return NULL;
}

int main (int argc, char** argv){
    return 1;
}
