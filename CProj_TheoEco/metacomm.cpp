// metacomm.cpp
// Build: see commands at bottom
// Notes:
//  - Exposes a C API for Python (ctypes)
//  - Uses your original dynamics, but takes inputs from Python
//  - OpenMP is optional; code compiles without it

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
  #include <omp.h>
#endif

#define MAX_SPECIES 128
#define MAX_NEI     8   // Moore 8-neighborhood

#ifdef _WIN32
  #define METACOMM_API extern "C" __declspec(dllexport)
#else
  #define METACOMM_API extern "C"
#endif

/* =================== Data Model =================== */
typedef struct {
  int n_sp;
  int rows, cols;
  int n_pch;
  // species params
  double mu[MAX_SPECIES];
  double sigma2[MAX_SPECIES];
  double c[MAX_SPECIES];
  double d[MAX_SPECIES];
  double m[MAX_SPECIES];
  double H[MAX_SPECIES][MAX_SPECIES];
  // patches (动态分配)
  double *e;   // size n_pch
  double *rho; // size n_pch
  // state (动态分配)
  double *R; // n_sp * n_pch
  double *P; // n_sp * n_pch
  // adjacency (动态分配，展平为 n_pch * MAX_NEI)
  int    *nbr_count;   // size n_pch
  int    *nbr_index;   // size n_pch * MAX_NEI, index[p*MAX_NEI + k]
  double *nbr_weight;  // size n_pch * MAX_NEI
  // scratch
  double *totalP;      // size n_pch
} Model;

static inline void recompute_R(Model *M) {
  const int n = M->n_sp;
  const int m = M->n_pch;
  for (int i = 0; i < n; ++i) {
    const double mu_i = M->mu[i];
    const double sig2 = M->sigma2[i];
    for (int p = 0; p < m; ++p) {
      double diff = M->e[p] - mu_i;
      M->R[i * m + p] = std::exp( -(diff*diff) / (2.0 * sig2) );
    }
  }
}

static inline void build_neighbors(Model *M) {
  const int rows = M->rows, cols = M->cols;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const int p = r * cols + c;
      int cnt = 0;
      for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
          if (dr == 0 && dc == 0) continue;
          int nr = r + dr, nc = c + dc;
          if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
          const int q = nr * cols + nc;
          const int off = p * MAX_NEI + cnt;
          M->nbr_index[off]  = q;
          // base weight by resistance (no d_i here; dispersal uses d_i)
          M->nbr_weight[off] = std::exp( -(M->rho[p] + M->rho[q]) / 2.0 );
          ++cnt;
        }
      }
      M->nbr_count[p] = cnt;
    }
  }
}

/* =================== Dynamics (from your code) =================== */
static void model_step(Model *M, double dt, double S) {
  const int n = M->n_sp;
  const int m = M->n_pch;

  // 1) total occupancy per patch
  #pragma omp parallel for
  for (int p = 0; p < m; ++p) M->totalP[p] = 0.0;

  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int p = 0; p < m; ++p) {
      #pragma omp atomic
      M->totalP[p] += M->P[i * m + p];
    }
  }

  // 2) species loop
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    const double ci = M->c[i];
    const double di = M->d[i];
    for (int p = 0; p < m; ++p) {
      const double P_ip = M->P[i * m + p];
      const double Ri   = M->R[i * m + p];

      double growth = ci * P_ip * Ri * (S - M->totalP[p]);

      // dispersal flux
      double flux = 0.0;
      const int cnt = M->nbr_count[p];
      const int base = p * MAX_NEI;
      for (int k = 0; k < cnt; ++k) {
        const int q = M->nbr_index[base + k];
        const double delta = M->P[i * m + q] - P_ip;
        // combine base resistance weight with species di
        const double w = std::exp( -(M->rho[p] + M->rho[q]) / (2.0 * di) );
        flux += di * w * delta;
      }

      double mortality = -M->m[i] * P_ip;

      // competitive displacement
      double competition = 0.0;
      for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        const double P_jp = M->P[j * m + p];
        competition += ci * P_ip * M->H[i][j] * P_jp - M->c[j] * P_jp * M->H[i][j] * P_ip;
      }
      competition *= Ri;

      double newP = P_ip + dt * (growth + flux + mortality + competition);
      if (newP < 0.0) newP = 0.0;
      M->P[i * m + p] = newP;
    }
  }
}

/* ========== helpers: free on failure / destruction ========== */
static void free_model(Model* M) {
  if (!M) return;
  free(M->e);
  free(M->rho);
  free(M->R);
  free(M->P);
  free(M->nbr_count);
  free(M->nbr_index);
  free(M->nbr_weight);
  free(M->totalP);
  free(M);
}

/* =================== Public API =================== */
METACOMM_API Model* mc_create_from_arrays(
  int rows, int cols, int n_sp,
  const double *e,   // size rows*cols
  const double *rho, // size rows*cols
  const double *mu, const double *sigma2, const double *c,
  const double *d,  const double *m,
  const double *H,   // size n_sp * n_sp, row-major H[i*n_sp + j]
  const double *P0   // size n_sp * (rows*cols); if NULL -> tiny seed
) {
  if (n_sp > MAX_SPECIES) return NULL;

  Model *M = (Model*)std::calloc(1, sizeof(Model));
  if (!M) return NULL;

  M->rows = rows; M->cols = cols; M->n_pch = rows * cols; M->n_sp = n_sp;

  // copy species arrays
  for (int i = 0; i < n_sp; ++i) {
    M->mu[i]     = mu[i];
    M->sigma2[i] = sigma2[i];
    M->c[i]      = c[i];
    M->d[i]      = d[i];
    M->m[i]      = m[i];
  }
  // H
  for (int i = 0; i < n_sp; ++i)
    for (int j = 0; j < n_sp; ++j)
      M->H[i][j] = H[i * n_sp + j];

  /* ---- 动态分配内存 ---- */
  const size_t npch   = (size_t)M->n_pch;
  const size_t ns_m   = (size_t)M->n_sp * (size_t)M->n_pch;
  const size_t adj_sz = (size_t)M->n_pch * (size_t)MAX_NEI;

  M->e   = (double*)std::malloc(sizeof(double) * npch);
  M->rho = (double*)std::malloc(sizeof(double) * npch);
  M->R   = (double*)std::calloc(ns_m, sizeof(double));
  M->P   = (double*)std::calloc(ns_m, sizeof(double));

  M->nbr_count  = (int*)   std::malloc(sizeof(int)    * npch);
  M->nbr_index  = (int*)   std::malloc(sizeof(int)    * adj_sz);
  M->nbr_weight = (double*)std::malloc(sizeof(double) * adj_sz);

  M->totalP     = (double*)std::malloc(sizeof(double) * npch);

  if (!M->e || !M->rho || !M->R || !M->P ||
      !M->nbr_count || !M->nbr_index || !M->nbr_weight || !M->totalP) {
    free_model(M);
    return NULL;
  }

  // patches
  std::memcpy(M->e,   e,   sizeof(double) * npch);
  std::memcpy(M->rho, rho, sizeof(double) * npch);

  // neighborhood
  build_neighbors(M);

  // resource matrix
  recompute_R(M);

  // initial occupancy
  if (P0) {
    std::memcpy(M->P, P0, sizeof(double) * ns_m);
  } else {
    // tiny seed if nothing provided
    for (size_t idx = 0; idx < ns_m; ++idx) M->P[idx] = 1e-3;
  }
  return M;
}

METACOMM_API void mc_run(Model *M, int steps, double dt, double S) {
  if (!M) return;
  for (int t = 0; t < steps; ++t) model_step(M, dt, S);
}

METACOMM_API void mc_get_P(Model *M, double *outP) {
  if (!M || !outP) return;
  std::memcpy(outP, M->P, sizeof(double) * (size_t)M->n_sp * (size_t)M->n_pch);
}

METACOMM_API void mc_free(Model *M) {
  free_model(M);
}

/* =================== Build notes =================== */
/*
Linux / macOS (Clang or GCC):
  g++ -O3 -fPIC -shared -o libmetacomm.so metacomm.cpp -fopenmp

macOS with Homebrew libomp (if needed):
  brew install libomp
  clang++ -O3 -fPIC -shared -o libmetacomm.dylib metacomm.cpp -Xpreprocessor -fopenmp -lomp

Windows (MSVC):
  cl /O2 /LD /openmp metacomm.cpp /Fe:metacomm.dll

Windows (MinGW):
  g++ -O3 -shared -o metacomm.dll metacomm.cpp -fopenmp -Wl,--out-implib,libmetacomm.a -Wl,--export-all-symbols -Wl,--enable-auto-import

If OpenMP isn’t available, you can drop the -fopenmp flag; the code compiles without it.
*/
