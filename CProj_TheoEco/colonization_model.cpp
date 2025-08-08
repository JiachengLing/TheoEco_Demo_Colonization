//
// Created by Jiacheng Ling on 2025-08-07.
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/* =============================================
 * Here is how you can add new ecological process
 * ---------------------------------------------
 * 1. Add relevant parameter in STEP 2 (for instance, birth rate, mortality etc.)
 * 2. Add this process in model_step (STEP 5)
 * =========================================== */

/* =============================================
 * STEP 1 -- DEFINE PATCH CAPABILITY
 * ===========================================*/
#define MAX_SPECIES 128
#define MAX_PATCHES 4096

/* =============================================
 * STEP 2 -- DEFINE DATA STRUCTURE
 * ============================================*/

typedef struct {
 /* ---------- BACKGROUND ----------- */
 int n_sp;               // number of species
 int rows, cols;         // Grid size
 int n_pch;              // number of patches = rows * cols

 /* ---------- SPECIES CONFIG --------- */
 double mu[MAX_SPECIES];       // trait z of species
 double sigma2[MAX_SPECIES];   // niche breadth of species
 double c[MAX_SPECIES];        // Colonization rate c_i
 double d[MAX_SPECIES];        // Dispersal rate d_i
 double m[MAX_SPECIES];        // Mortality rate m_i

 /* ----------- PATCH CONFIG ---------- */
 double e[MAX_PATCHES];        // Environmental condition e_j
 double rho[MAX_PATCHES];      // Dispersal resistance  rho_j

 /* ---------- DYNAMICS --------- */
 double *R;                    // SPECIES × PATCHES resource matrix
 double *P;                    // SPECIES × PATCHES occupancy matrix

 /* ---------- ADJACENT INFO ---------- */
 int    nbr_count[MAX_PATCHES];        // neighbor count per patch
 int    nbr_index[MAX_PATCHES][8];     // indices of neighbors (Moore 8)
 double nbr_weight[MAX_PATCHES][8];    // base weight exp(-(rho_p+rho_q)/2)
} Model;

/* ======================================
 * STEP 3 -- GENERATE RANDOM VALUES
 * ==================================== */
static double urand(double a, double b) {
 return a + (b - a) * (double)rand() / RAND_MAX;
}

/* ======================================
 * STEP 4 -- CREATE MODEL
 * =================================== */
Model *model_create(int rows, int cols, int n_sp) {
 if (rows * cols > MAX_PATCHES || n_sp > MAX_SPECIES) {
  fprintf(stderr, "[ERROR] Exceeds MAX_PATCHES or MAX_SPECIES.\n");
  return NULL;
 }

 Model *M = (Model*)calloc(1, sizeof(Model));
 M->rows  = rows;
 M->cols  = cols;
 M->n_pch = rows * cols;
 M->n_sp  = n_sp;

 /* allocate matrices */
 M->R = (double*)calloc((size_t)n_sp * M->n_pch, sizeof(double));
 M->P = (double*)calloc((size_t)n_sp * M->n_pch, sizeof(double));

 /* ----- Randomize species ----- */
 for (int i = 0; i < n_sp; ++i) {
  M->mu[i]     = urand(0.0, 1.0);
  M->sigma2[i] = urand(0.01, 0.1);
  M->c[i]      = urand(0.2, 0.6);
  M->d[i]      = urand(0.01, 0.05);
  M->m[i]    = urand(0.01, 0.6);
 }

 /* -----  Randomize patches  ----- */
 for (int p = 0; p < M->n_pch; ++p) {
  M->e[p]   = urand(0.0, 1.0);
  M->rho[p] = urand(0.5, 3.0);
 }

 /* ---- Calculate resource matrix & seed occupancy -----*/
 for (int i = 0; i < n_sp; ++i) {
  for (int p = 0; p < M->n_pch; ++p) {
   double diff = M->e[p] - M->mu[i];
   M->R[i * M->n_pch + p] = exp( -(diff*diff) / (2.0 * M->sigma2[i]) );
   M->P[i * M->n_pch + p] = urand(0.0, 0.01);    // tiny seed
  }
 }

 /* ----- Construct neighbor table -----*/
 for (int r = 0; r < rows; ++r) {
  for (int c = 0; c < cols; ++c) {
   int p = r * cols + c;
   int cnt = 0;
   for (int dr = -1; dr <= 1; ++dr)
    for (int dc = -1; dc <= 1; ++dc) {
     if (dr == 0 && dc == 0) continue;
     int nr = r + dr, nc = c + dc;
     if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
     int q = nr * cols + nc;
     M->nbr_index[p][cnt]  = q;
     M->nbr_weight[p][cnt] = exp( -(M->rho[p] + M->rho[q]) / 2.0 );
     ++cnt;
    }
   M->nbr_count[p] = cnt;
  }
 }
 return M;
}

/* ======================================
 * STEP 5 -- ONE‐STEP UPDATE
 * ==================================== */
void model_step(Model *M, double dt, double S) {
 static double totalP[MAX_PATCHES];
 int n = M->n_sp;
 int m = M->n_pch;

 /* 5.1 total occupancy per patch */
 for (int p = 0; p < m; ++p) totalP[p] = 0.0;
 #pragma omp parallel for
 for (int i = 0; i < n; ++i)
  for (int p = 0; p < m; ++p)
   #pragma omp atomic
   totalP[p] += M->P[i * m + p];

 /* 5.2 loop through species */
 #pragma omp parallel for
 for (int i = 0; i < n; ++i) {
  double ci = M->c[i];
  double di = M->d[i];
  for (int p = 0; p < m; ++p) {
   double P_ip = M->P[i * m + p];
   double growth = ci * P_ip * M->R[i * m + p] * (S - totalP[p]); // growth

   double flux = 0.0;                                             // dispersal
   int cnt = M->nbr_count[p];
   for (int k = 0; k < cnt; ++k) {
    int q = M->nbr_index[p][k];
    double delta = M->P[i * m + q] - P_ip;
    double w = exp( -(M->rho[p] + M->rho[q]) / (2.0 * di) );
    flux += di * w * delta;
   }

   double mortality = -M->m[i] * P_ip;                           // For example, define mortality here

   double newP = P_ip + dt * (growth + flux + mortality);         // final occupancy
   if (newP < 0.0) newP = 0.0;
   M->P[i * m + p] = newP;
  }
 }
}

/* --- STEP 6 REPORT ---*/
static void print_richness(Model *M) {
 int alive = 0;
 for (int i = 0; i < M->n_sp; ++i) {
  for (int p = 0; p < M->n_pch; ++p) {
   if (M->P[i * M->n_pch + p] > 1e-6) { ++alive; break; }
  }
 }
 printf("Present species richness: %d\n", alive);
}




int main(void) {
 srand(42);
 Model *M = model_create(4, 4, 20); // 4×4 grid, 20 species
 if (!M) return 1;

 const double dt = 0.01, S = 1.0;
 for (int t = 0; t < 100000; ++t) {
  model_step(M, dt, S);
  if (t % 1000 == 0) print_richness(M);
 }

 /* dump final occupancy */
 FILE *fp = fopen("final_P.csv", "w");
 for (int i = 0; i < M->n_sp; ++i) {
  for (int p = 0; p < M->n_pch; ++p) {
   fprintf(fp, "%g%c", M->P[i * M->n_pch + p], p == M->n_pch-1 ? '\n' : ',');
  }
 }
 fclose(fp);

 free(M->R); free(M->P); free(M);
 return 0;
}
