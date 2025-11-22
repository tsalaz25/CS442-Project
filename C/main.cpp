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
			A[i * N + j] = (i == j) ? 2.0 : 1.0;
			B[i * N + j] = (i == j) ? 1.0 : 0.5;
		}
	}
}

static void local_accum(const double *A, const double *B, double *C, int rows, int K, int cols){
	for (int i = 0; i < rows; i++){
		for (int k = 0; k < cols; k++){
			double sum = 0.0;
			for (int j = 0; j < K; j++){
				sum += A[i * K + j] * B[j * cols + k];
			}

			C[i * cols + k] += sum;
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
			double *src = A + r0 * n;

			if (p == 0){
				std::copy(src, src + BK * n, A_local.begin());
			} else {
				MPI_Send(src, BK * n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
			}
		}	
		//Send full B to other ranks
		std::copy(B, B + n * n, B_full.begin());
		for (int p = 1; p < size; p++){
			MPI_Send(B, n * n, MPI_DOUBLE, p , 1, MPI_COMM_WORLD);
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

		for (int ii = 0; ii < BK; ii += BK){
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

			for (int p = 1; p < size; p++){
					int r0 = p * BK;
					MPI_Recv(C + r0 * n, BK * n, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
	std::copy(B, B + block * block, B_work.begin());

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
	std::copy(B, B + block * block, B_curr.begin());

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

	for (int step = 0; step < q; step++){
		local_accum(A_curr.data(), B_curr.data(), C, block, block, block);

		MPI_Sendrecv_replace(A_curr.data(), block * block, MPI_DOUBLE, left, 3, right, 3,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Sendrecv_replace(B_curr.data(), block * block, MPI_DOUBLE, up, 4, down, 4,
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

	const int BK = n/size;

	std::vector<double> A_local (BK * n);
	std::vector<double> B_full (n * n);
	std::vector<double> C_local (BK * n);

	//Window for A and B, Roots exposed, others with size 0
	MPI_Win winA, winB;
	if (rank == 0){
		MPI_Win_create(A, n * n * sizeof(double), sizeof(double),
					   MPI_INFO_NULL, MPI_COMM_WORLD, &winA);
		MPI_Win_create(B, n * n * sizeof(double), sizeof(double),
					   MPI_INFO_NULL, MPI_COMM_WORLD, &winB);
	} else {
		MPI_Win_create(nullptr, 0, sizeof(double),
						MPI_INFO_NULL, MPI_COMM_WORLD, &winA);
		MPI_Win_create(nullptr, 0, sizeof(double),
						MPI_INFO_NULL, MPI_COMM_WORLD, &winB);
	}

	//Each Rank Pulls Rows of A and B from rank 0
	MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, winA);
	MPI_Get(A_local.data(), BK * n, MPI_DOUBLE, 0, 
			rank * BK * n, BK * n, MPI_DOUBLE, winA);
	MPI_Win_unlock(0, winA);

	MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, winB);
	MPI_Get(B_full.data(), n * n, MPI_DOUBLE, 0,
			0, n * n, MPI_DOUBLE, winB);
	MPI_Win_unlock(0, winB);

	//Every Rank now has A(BK * n) and B(n*n)
	for (int iter = 0; iter < n_iter; iter++){
		std::fill (C_local.begin(), C_local.end(), 0.0);

		//Same Blocked structure from before
		for (int ii = 0; ii < BK; ii += BK){
			int i_end = (ii + BK < BK) ? (ii + BK) : BK;

			for (int kk = 0; kk < n; kk += BK){
				int k_end = (kk + BK < n) ? (kk + BK) : n;

				for (int jj = 0; jj < n; jj += BK){
					int j_end = (jj + BK < n) ? (jj + BK) : n;
					
					for (int i = ii; i < i_end; i++){
						double *c_row = &C_local[i*n];

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

	MPI_Win_free(&winA);
	MPI_Win_free(&winB);

	//Gather Local C to Rank 0
	if (rank == 0){
		std::copy(C_local.begin(), C_local.end(), C);
		for (int p = 1; p < size; p++){
			int r0 = p * BK;
			MPI_Recv(C + r0 * n, BK * n, MPI_DOUBLE, p , 2,
					 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	} else {
		MPI_Send(C_local.data(), BK * n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
	}


}

void rma_fox (double* A, double* B, double* C, int n, int sq_num_procs, int rank_row, int rank_col) {
	return;
}

void rma_cannon(double* A, double* B, double* C,int n, int sq_num_procs, int rank_row, int rank_col){
	int rank, num_procs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	int q = sq_num_procs;
	int size = n * n; //Elems per block

	for (int i = 0; i < size; i++){
		C[i] = 0.0;
	}

	//Expose A and B Blocks in Windows
	MPI_Win winA, winB;
	MPI_Win_create(A, size * sizeof(double), sizeof(double),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &winA);
	MPI_Win_create(B, size * sizeof(double), sizeof(double),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &winB);

	//Temp Buffers
	double *buf_A = new double[size];
	double *buf_B = new double[size];

	//Multiply Accum
	auto mul_acc = [&](const double *AA, const double *BB){
		for(int i = 0; i < n; i++){
			for (int k = 0; k < n; k++){
				const double aik = AA[i * n + k];
				for (int j = 0; j < n; j++){
					C[i * n + j] += aik * BB[k * n + j];
				}
			}
		}
	};

	//For each K, pull A[i,k] and B[k,j]
	for (int k = 0; k < q; k++){
		int rankA = coords_to_rank(rank_row, k, q);
		int rankB = coords_to_rank(k, rank_col, q);

		//fetch A[ik] block
		MPI_Win_lock(MPI_LOCK_SHARED, rankA, 0, winA);
		MPI_Get(buf_A, size, MPI_DOUBLE,
                rankA, 0, size, MPI_DOUBLE, winA);
		MPI_Win_unlock(rankA, winA);

		//fetch B[kj]
		MPI_Win_lock(MPI_LOCK_SHARED, rankB, 0, winB);
		MPI_Get(buf_B, size, MPI_DOUBLE,
                rankB, 0, size, MPI_DOUBLE, winB);
		MPI_Win_unlock(rankB, winB);

		mul_acc(buf_A, buf_B);
	}

	delete[] buf_A;
    delete[] buf_B;

    MPI_Win_free(&winA);
    MPI_Win_free(&winB);
}


int main (int argc, char** argv){
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	if (argc < 2) {
		if (rank == 0) { fprintf(stderr, "usage ./matmul N\n"); }
		MPI_Finalize();
		return 1;
	}

	int N = atoi(argv[1]);
	if (N % size != 0){
		if (rank == 0) { fprintf(stderr, "N mut be Divisible by P\n"); }
		MPI_Finalize();
        return 1;
	}

	int q = (int)std::sqrt(size);
	if (q * q != size){
		if (rank == 0) { fprintf(stderr, "Fox/Cannon require perfect Squares");}
		MPI_Finalize();
        return 1;
	}

	int BK = N/q;

	//Allocate Matricies
	std::vector<double> A_global, B_global, C_global;
	if (rank == 0){
		A_global.resize(N*N);
		B_global.resize(N*N);
		C_global.resize(N*N);

		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++){
				A_global[i*N + j] = (i == j ? 2.0 : 1.0);
				B_global[i*N + j] = (i == j ? 1.0 : 0.5);
			}
		}
	}

	//Blocking for Cannons/Fox
	std::vector<double> A_blk(BK * BK);
	std::vector<double> B_blk(BK * BK);
	std::vector<double> C_blk(BK * BK, 0.0);

	//compute coords in qxq grid
	int rank_row = rank/q;
	int rank_col = rank % q;

	//Scatter Blocks
	if (rank == 0 ){

		for (int p = 0; p < size; p++){
			int rr = p / q;
			int cc = p % q;

			for (int i = 0 ; i < BK; i++){
				for (int j = 0; j < BK; j++){
					int gi = rr * BK + i;
					int gj = cc * BK + j;

					double Aval = A_global[gi*N + gj];
					double Bval = B_global[gi*N + gj];

					if (p == 0){
						A_blk[i*BK + j] = Aval;
						B_blk[i*BK + j] = Bval;
					} else {
						MPI_Send(&Aval, 1, MPI_DOUBLE, p, 10, MPI_COMM_WORLD);
						MPI_Send(&Bval, 1 ,MPI_DOUBLE, p, 11, MPI_COMM_WORLD);
					}
				}
			}
		}
	} else {
		for (int i = 0; i < BK; i++){
			for (int j = 0; j < BK; j++){
				MPI_Recv(&A_blk[i*BK + j], 1, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(&B_blk[i*BK + j], 1, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);



	//Timing Algo
	auto time_it = [&](const char* name, auto fn){
		MPI_Barrier(MPI_COMM_WORLD);
		double t0 = MPI_Wtime();
		fn();
		MPI_Barrier(MPI_COMM_WORLD);
		double t1 = MPI_Wtime();

		if (rank == 0){
			printf("%-16s %8.6f sec\n", name, t1 - t0);
		}
	};

	//Run Algos
	time_it("MPI Blocked", [&]() {
		if (rank == 0) std::fill(C_global.begin(), C_global.end(), 0.0);
		blocked_matmat(N, rank == 0 ? A_global.data() : nullptr, 
						  rank == 0 ? B_global.data() : nullptr,
						  rank == 0 ? C_global.data() : nullptr, 1); //1 iteration since large sizes
	});

	time_it("MPI Fox", [&](){
		std::fill(C_blk.begin(), C_blk.end(), 0.0);
		fox_matmat(A_blk.data(), B_blk.data(), C_blk.data(), BK, q, rank_row, rank_col);
	});

	time_it("MPI Cannon", [&](){
		std::fill(C_blk.begin(), C_blk.end(), 0.0);
		cannon_matmat(A_blk.data(), B_blk.data(), C_blk.data(), BK, q, rank_row, rank_col);
	});

	time_it("RMA Blocked", [&](){
		if (rank == 0) std::fill(C_global.begin(), C_global.end(), 0.0);
		rma_blocked(N, rank == 0 ? A_global.data() : nullptr,
					   rank == 0 ? B_global.data() : nullptr,
					   rank == 0 ? C_global.data() : nullptr, 1);
	});

	time_it("RMA Cannon", [&](){
		std::fill(C_blk.begin(), C_blk.end(), 0.0);
		rma_cannon(A_blk.data(), B_blk.data(), C_blk.data(), BK, q, rank_row, rank_col);		
	});

	if (rank == 0) {
		printf("\nDone\n");
	}

	MPI_Finalize();
	return 0;
}

