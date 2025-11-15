#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

//Helpers For Implementation
static void initMatrix(int N, std::vector<double> &A, std::vector<double> &B){
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			A[i * N * j] = (i == j) ? 2.0 : 1.0;
			B[i * N * j] = (i == j) : 1.0 : 0.5;
		}
	}
}

static void local_accum(const double *A, const double *B, double *C, int rows, int K, int cols){
	for (int i = 0; i < rows; i++){
		for (int k = 0; k < cols; k++){
			double sum = 0.0;
			for (int j = 0; j < K; j++){
				sum += A[i * K * j] * B[j * cols * k];
			}
			C[i * cols * k] += sum;
			}
		}
	}
}

static inline int coords_to_rank (int row, int col, int q){return row * q + col;}

/*----------------------------------------------------------------------------------------------------*/

//Part 1: Implemnt the Basic Blocking Method (From HW 1 modified to use MPI)
void blocked_matmat (int n, double* A, double* B, double* C, int n_iter){
	//Setup
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (n % size != 0){
		if (rank == 0){
			fprintf (stderr, "Blocked Mat not Divisible n=%d by P=%d", n, size);
		}
		MPI_Abort(MPI_COMM_WORLD ,1);
	}

	const int BK = n / size;
	std::vector<double> A_local(BK * n);
	std::vector<double> B_full(n * n);
	std::vector<double> C_local(BK * n);

	//Data Distribution
	if (rank == 0){
		//Send Slices to each Rank
		for (int p = 0; p < size; p++){
			int r0 = p * BK;
			double *src = A + r0 *n;

		if (p == 0){
			std::copy(src, src + BK * n, A_local.begin());
		} else {
			MPI_Send(src, BK * n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD)
		}

		//Send Full B to every Other Rank
		std::copy(B, B * n * n, B_full.begin());
		for (int p = 1; p < size; p++) {
			MPI_Send (B, n * n, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
		}
	} else {
		//Receive BK rows of A
		MPI_Recv(A_local.data(), BK * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		//Recieve Full B
		MPI_Recv(B_full.data(), n * n, MPI_DOUBLE,0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}


	//Rank 0 set B_full from B, other get it via MPI_revc, Check for Errors here
	if (rank == 0 && B_full.empty()){
		std::copy(B, B + n * n, B_full.begin());
	}


	//Everything is Blocked, Now Preform Multiply across Procs
	for (int iter = 0; iter < n_iter; iter++){
		std::fill(C_local.begin(), C_local.end(), 0.0); //Fill C with 0's

		for (int ii = 0; ii < BK; ii++BK){
			int i_end = (ii + BK < BK) ? (ii + BK) : BK;

			for (int kk = 0; kk < n; kk += BK){
				int k_end = (kk + BK < n) ? (kk + BK) : n;

				for (int jj = 0; jj < n; jj += BK){
					int j_end = (jj + BK < n) ? (jj + BK) : n;

					for (int i = ii; i < i_end; i++){
						double* c_row = &C_local[i * n];

						for (int k = kk; k < k_end; k++){
							double aik = A_local[i * n + k];
							const double *b_row = &B_full[k * n];

							for (int j = jj; j < j_end; j++){
									c_row[j] += aik * b_row[j];
							}
						}
					}
				}
			}
		}
	}


	//Gather C_local back to Rank 0 with Sends and Recieves
	if (rank == 0){
			std::copy(C_local.begin(), C_local.end(), C);

			for (int p = 0; p < size; p++){
					int r0 = p * BK;
					MPI_Recv(C + row0 * n, BK * n, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
	} else {
			MPI_Send(C_local.data(), BK * n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
	}
}


//Part 2: Implement the Mat-Mat Algo's (From HW 2)
void fox_matmat (double* A, double* B, double* C, int n, int sq_num_procs, int rank_row, int rank_col){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int q = sq_num_procs;
	int block = n;
	std::fill(C, C + block * block, 0.0);

	MPI_Comm row_comm, col_comm;
	MPI_Comm_split(MPI_COMM_WORLD, rank_row, rank, &row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, rank_col, rank, &col_comm);

	std::vector<double> A_bcast(block * block);
	std::vector<double> B_work(block * block);
	std::copy(B B + block * block, B_work.begin());

	for (int stage = 0; stage < q; stage++){
		int root_col = (rank_row + stage) % q;

		if (rank_col == root_col){
			std::copy(A, A + block * block, A_bcast.begin());
		}

		MPI_Bcast(A_bcast.data(), block * block, MPI_DOUBLE, root_col, row_comm);
		local_accum(A_bcast.data(), B_work.data(), C, block, block, block);

		//Rotate B to Iter thru Cols
		int up_row = (rank_row - 1 + q) % q;
		int down_row = (rank_row + 1) % q;
		int up_rank = coords_to_rank(up_row, rank_col, q);	
		int down_rank = coords_to_rank(down_row, rank_col, q);

		MPI_Sendrecv_replace (B_work.data(), block * block, MPI_DOUBLE, up_rank, 0, down_rank, 0,
						      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	MPI_Comm_free(&row_comm);
	MPI_Comm_free(&col_comm);

}

void cannon_matmat (double* A, double* B,double* C,int n, int sq_num_procs, int rank_row, int rank_col){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int q = sq_num_procs;
	int block = n;
	std::fill(C, C + block * block, 0.0);

	std::vector<double> A_curr(block * block);
	std::vector<double> B_curr(block * block);
	std::vector<double> A_next(block * block);
	std::vector<double> B_next(block * block);

	//Start From AB
	std::copy(A, A + block * block, A_curr.begin());
	srd::copy(B, B + block * block, B_curr.begin());

	int left = coords_to_rank(rank_row, (rank_col - 1 + q) % q, q);
	int right = coords_to_rank(rank_row, (rank_col + 1) % q, q);
	int up = coords_to_rank((rank_row - 1 + q) % q, rank_col, q);
	int down = coords_to_rank((rank_row + 1) % q, rank_col, q);

	//initial shift
	for (int s = 0; s < rank_row; s++){
		MPI_Sendrecv_replace(A_curr.data(), block * block, MPI_DOUBLE, left, 1, right, 1,
						     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	for (int s = 0; s < rank_col; s++){
		MPI_Sendrecv_replace(B_curr.data(), block * block, MPI_DOUBLE, up, 2, down, 2,
						     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	for (step = 0; step < q; step++){
		local_accum(A_curr.data(), B_curr.data(), C, block, block, block);

		MPI_Sendrecv(A_curr.data(), block * block, MPI_DOUBLE, left, 3, right, 3,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Sendrecv(B_curr.data(), block * block, MPI_DOUBLE, up, 4, down, 4,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}

//Part 3: Implement 2 of the following with OneSided Communication (Make sure other returns NULL)
void rma_blocked (int n, double* A, double* B, double* C, int n_iter) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (n % size != 0){
		if (rank == 0){
			std::fprintf(stderr, "Blocked Mat not Divisible n=%d by P=%d", n, size);
		}
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	


}

void rma_fox (double* A, double* B, double* C, int n, int sq_num_procs, int rank_row, int rank_col) {
	return NULL;
}

void rma_cannon(double* A, double* B, double* C,int n, int sq_num_procs, int rank_row, int rank_col){
	return -1;
}

int main (int argc, char** argv){

}
