# Sudbury-Paper

# Data and Scripts for Impact and Crystallization Modeling of the Sudbury Basin

## Overview
This archive contains input files and analysis scripts used in the study:

**Impact and Crystallization Modeling of the Sudbury Basin and its Implications for a Hadean Crust**  
Litza et al. (2026)

The dataset supports iSALE impact simulations and alphaMELTS2 crystallization modeling
applied to the Sudbury Igneous Complex.

---

## iSALE Input Files
There are 16 iSALE input (`.inp`) files in total.  
Asteroid and material input files are paired for four target environments
(Environments A–D) and for both granite and ice impactors.

### Granite Impactor
1. **Environment A**  
   - `Granite_Env_A_asteroid.inp`  
   - `Granite_Env_A_material.inp`

2. **Environment B**  
   - `Granite_Env_B_asteroid.inp`  
   - `Granite_Env_B_material.inp`

3. **Environment C**  
   - `Granite_Env_C_asteroid.inp`  
   - `Granite_Env_C_material.inp`

4. **Environment D**  
   - `Granite_Env_D_asteroid.inp`  
   - `Granite_Env_D_material.inp`

### Ice Impactor
5. **Environment A**  
   - `Ice_Env_A_asteroid.inp`  
   - `Ice_Env_A_material.inp`

6. **Environment B**  
   - `Ice_Env_B_asteroid.inp`  
   - `Ice_Env_B_material.inp`

7. **Environment C**  
   - `Ice_Env_C_asteroid.inp`  
   - `Ice_Env_C_material.inp`

8. **Environment D**  
   - `Ice_Env_D_asteroid.inp`  
   - `Ice_Env_D_material.inp`

---

## MELTS Input Files
There are two input files used for alphaMELTS2 crystallization modeling:

1. `LGC_composition.melts`  
   Establishes major oxide compositions and run parameters, including temperature
   step, initial and final temperature, and crystallization mode.

2. `MELT_batch.txt`  
   Defines the order of operations and specifies the algorithm used by the
   MELTS front end.

---

## iSALE Analysis Scripts
There are two Python scripts used for post-processing iSALE simulations:

1. `diameter_calculator.py`  
   Measures final crater diameter from completed iSALE simulations run in
   DEFAULT mode.

2. `meltsheet_calculator.py`  
   Calculates final melt sheet volume from completed iSALE simulations run in
   DEFAULT mode.

---

## MELTS Analysis Script
There is one Python script used for analyzing MELTS results:

1. `Aitchison_calculator.py`  
   Calculates Aitchison distance from completed MELTS simulations.

---

## Notes
These files are intended to support the results presented in the associated
manuscript. Detailed model descriptions and assumptions are provided in the paper.
