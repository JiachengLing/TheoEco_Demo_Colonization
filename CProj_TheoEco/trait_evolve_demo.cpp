//
// Created by Jiacheng Ling on 2025-08-08.
//
/* =============================================
 * STEP 1 -- DEFINE PATCH CAPABILITY
 * ===========================================*/

#define MAX_SPECIES 2

/* =============================================
 * STEP 2 -- DEFINE DATA STRUCTURE
 * ============================================*/


typedef struct {

 /* ------ BACKGROUND ----- */
 int n_sp;

 double mu[MAX_SPECIES]; // trait z of species
 
} Model;