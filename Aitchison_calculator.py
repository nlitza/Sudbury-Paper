# As of 17:00, JUNE 11, 2024!
# UNDER CONSTRUCTION

import pandas as pd
import numpy as np
import time

start_time = time.time()

# Begin by loading in the compositions of the layers you want to compare to
#          [SiO2=0, TiO2=1, Al2O3=2, Fe2O3=3, Cr2O3=4, FeO=5, MnO=6, MgO=7, NiO=8, CoO=9, CaO=10, Na2O=11, K2O=12,  P2O5=13]
dry_MNR  = [56.20,  0.60,   10.31,   3.17,    0,       8.36,  0.18,  13.42, 0.06,  0.01,  4.43,   1.91,    1.26,    0.09]
dry_FNR  = [57.98,  0.50,   17.33,   2.49,    0,       4.73,  0.11,  5.18,  0.01,  0.01,  6.88,   3.22,    1.46,    0.1]
dry_QGAB = [55.55,  1.73,   14.60,   4.54,    0,       7.18,  0.14,  3.56,  0.01,  0.01,  6.78,   3.90,    1.42,    0.58]# H2O   CO2
wet_GRAN = [67.83,  0.86,   12.74,   2.99,    0,       3.23,  0.09,  1.19,  0.01,  0.01,  1.74,   3.69,    3.47,    0.22,  1.86, 0.07]
sigma_n = len(dry_MNR) # this is the amount of oxides you are comparing to. Right now this is 14. 

#######################################################################
### --------------------------- Volumes ----------------------------###
#######################################################################
# This section processes the 'Phase_vol_tbl.txt' file and determines the valid column 
# iterations from the MELTS output based on specified volume ratios for different layers.
#
# The first layer, Mafic Norite (MNR), should form first and constitute 2.5% - 7.5% of the 
# total volume. The code calculates the cumulative volume from the beginning of the simulation 
# and identifies which columns meet these criteria. For instance, 2.5% - 7.5% of the total 
# volume might be reached between the 3rd and 5th temperature steps.
#
# The next layer, Felsic Norite (FNR), begins forming as soon as the MNR layer finishes. The 
# code loops through the temperature steps starting from 4th (3rd+1) to 6th (5th+1) step to 
# build the FNR layer until it reaches its target volume of 17.5% - 22.5%.
#
# This process continues similarly for the Quartz Gabbro (QGAB) and Granophyric (GRAN) layers.
# The GRAN composition represents the remaining magma composition after QGAB has formed.

#####################################
## Layer Constraints Based on Vol% ##
#####################################
bounds = {'MNR' : {'lower': 2.5,  'upper': 7.5},  'FNR' : {'lower': 17.5, 'upper': 22.5},
          'QGAB': {'lower': 12.5, 'upper': 17.5}, 'GRAN': {'lower': 50,  'upper': 70} }
# These values assume 2.5 < MNR < 7.5 by wt%, etc. 

# Load in the Phase_vol_tbl.txt file and begin calculating volumes
volume_calculator = pd.DataFrame()
volume = pd.read_csv('Phase_vol_tbl.txt', delimiter=' ', skiprows=1) # delete top row of the file
volume = volume.iloc[:, :-1].drop(volume.columns[[1]], axis=1) # remove the NaN column
liquid1_index = volume.columns.get_loc("liquid1") # Get the index of the "liquid1" column
fluid1_column = volume.pop("fluid1") # Get the "fluid1" column
volume.insert(liquid1_index , "fluid1", fluid1_column) # Moving fluid1 to be to the left of liquid1
volume['Vol All Solids'] = volume.iloc[:, (liquid1_index+2):].sum(axis=1) # Total volume of all solids
volume_copy = volume.copy() # Need a copy since we need the row before solids form
volume = volume[volume['Vol All Solids'] != 0] # delete all of the rows where NO solid has formed 
rolling_volume = volume.copy()
rolling_volume.iloc[:,4:] = rolling_volume.iloc[:, 4:].cumsum() # Rolling mass of Phase Mass
index_position = rolling_volume['index'].iloc[0]
starting_liquid = volume_copy['liquid1'].iloc[index_position - 2] # Starting liquid volume

# Calculate the total volume (values in cubic cm, cc)
# Phase_vol_tbl.txt shows the volume of solids that have formed, remaining magma volume called 'liquid1'
# and a mix of CO2+H2O called 'fluid1'. Volumes reported by the solids are from each temperature step,
# liquid1 and fluid1 are reported values for the entire system. Volume of liquid1 and fluid1 will actively
# change over the course of the simulation. Some H2O will be partitioned into solids (i.e., Apatite)
# Crystallization will shrink the magma by about 90% 
# ------ Thus, the volume is calculated as such: 
# 1) Starting magma volume is the value in liquid1 in the row prior to phases forming, typically ~40cc
# 2) Total magma volume crystallized is the difference between the starting magma volume and the volume of 
#    liquid1 at the end of the simulation
# 3) Simulation will may crash with remaining magma remaining to be crystallized. This magma will contribute to 
#    the final volume. Remaining magma is multiplied by a fudge factor called 'volume_shrink', typically ~0.9, 
#     and is calculated by seeing ratio of volumes from total solid formed compared to total magma used. 
# 4) The final volume is the sum of the total volume formed plus the remaining volume (which is multiplied by the 
#    shrink fudge factor). The final volume is expected to be 90% of the starting value.
# 5) Volume from fluid1 DOES NOT contribute to the final volume, its mass is negligible to the system.

starting_liquid = volume_copy['liquid1'].iloc[index_position - 2] # Starting liquid volume
del volume_copy # Delete the copy to free memory
liquid_used = starting_liquid - rolling_volume.iloc[-1,3] # Liquid used in the process
volume_shrink = rolling_volume.iloc[-1,-1] / liquid_used  # Shrinkage factor for volume
final_volume = (volume_shrink*rolling_volume['liquid1'].iloc[-1]) + rolling_volume.iloc[-1,-1]

rolling_volume['Vol %'] = rolling_volume['Vol All Solids'].iloc[:] / final_volume * 100 # Total volume by each step
rolling_volume['Remain Vol %'] = 100 - rolling_volume['Vol %'].iloc[:] 

# Apply the volume% filter to the FNR 
vol_percent_filtered = rolling_volume[(rolling_volume['Vol %'] > bounds['MNR']['lower']) 
                                    & (rolling_volume['Vol %'] < bounds['MNR']['upper'])]
i_values = vol_percent_filtered['index'].to_numpy() - rolling_volume['index'].iloc[0]
# i_values is all of the indexes where MNR can formed based on the volume constraints. 

#######################################################################
### -------------------------- Mass Setup --------------------------###
#######################################################################

# Load in the Solid Composition Table file
# Solid_comp_tbl.txt: mass and bulk composition of the solid residue in wt% oxides
df = pd.read_csv('Solid_comp_tbl.txt', delimiter=' ', skiprows=1)
df = df.iloc[:, :-1].drop(df.columns[[1]], axis=1) # Remove the first row and pressure column
df.replace('---', pd.NA, inplace=True)
df = df.fillna(0)
df = df.apply(pd.to_numeric, errors='coerce')

# Sum all non-H2O&CO2 oxides for each temperature step, these values are set to columns 3-17 
df['Solid Sum%'] = df.iloc[:, 3:17].sum(axis=1) # wt% of oxides that are not H2O and CO2

# There is some H2O that ends up in Apatite, this value is negligble

df['Fluid Sum%'] = df.iloc[:, 17:19].sum(axis=1) # Sum of H2O and CO2
df['Solid Mass'] = df.iloc[:, 2] * df.iloc[:, -2] / 100 # Mass of all oxides at EACH temp step
df = df[df['Solid Mass'] != 0] # delete all of the columns where NO solid has formed 
df.reset_index(drop=True, inplace=True)

# Finding mass of each oxide at each step
denominator_cols = df.columns[3:17]
denominator = df['Solid Sum%'] / 100
df_normalized = df[denominator_cols].div(denominator, axis=0)
df_normalized = df_normalized.where(denominator != 0, np.nan)

# Make a new dataframe that calculates the mass of each oxide at each step
df_mass_by_step = pd.DataFrame()
df_mass_by_step['index'] = df['index']
df_mass_by_step[df.columns[-1]] = df.iloc[:, -1]
df_mass_by_step[df_normalized.columns] = df_normalized.mul(df['Solid Mass'].values / 100, axis=0)
# Each of these columns represent an oxide and how many grams crystallizing and fractionating at 
# each temperature step. Next step is to find the cumulative mass and its subsequent wt%

del df # free up some memory
del df_normalized

df_rolling_mass = df_mass_by_step.copy()
df_rolling_mass.iloc[:,1:] = df_rolling_mass.iloc[:, 1:].cumsum() # cumulative some of mass at each step
df_rolling_wt_per = df_rolling_mass.copy()
df_rolling_wt_per.iloc[:, 2:] = df_rolling_wt_per.iloc[:, 2:] / df_rolling_wt_per.iloc[:, [1]].values * 100 
# The Rolling Weight Percent table represents the bulk composition of the formed solids up to a specific point

# Phase_mass_tbl.txt: the masses of all phases present in more compact format
df_solids = pd.read_csv('Phase_mass_tbl.txt', delimiter=' ', skiprows=1)
df_solids = df_solids.iloc[:, :-1].drop(df_solids.columns[[1]], axis=1) #remove the first row and the column which spits out the pressure
liquid1_index = df_solids.columns.get_loc("liquid1") # Get the index of the "liquid1" column
fluid1_column = df_solids.pop("fluid1") # Get the "fluid1" column
df_solids.insert(liquid1_index + 1, "fluid1", fluid1_column) # I'm just moving fluid1 to be to the left of liquid1
df_solids['Mass All Solids'] = df_solids.iloc[:, 4:].sum(axis=1) # Total mass of all solids
df_solids = df_solids[df_solids['Mass All Solids'] != 0] # delete all of the columns where NO solid has formed 

df_PM_rolling_mass = df_solids.copy() # Rolling mass of the Phases
df_PM_rolling_mass.iloc[:,4:] = df_PM_rolling_mass.iloc[:, 4:].cumsum() # Rolling mass of Phase Mass
df_PM_rolling_wt_per = df_PM_rolling_mass.copy()
df_PM_rolling_wt_per.iloc[:, 4:-1] = df_PM_rolling_wt_per.iloc[:, 4:-1] / df_PM_rolling_wt_per.iloc[:, [-1]].values * 100

df_PM_step_wt_per = df_solids.copy()
df_PM_step_wt_per.iloc[:, 4:-1] = df_PM_step_wt_per.iloc[:, 4:-1] / df_solids.iloc[:, [-1]].values * 100

# Now we know the weight percentage from each of the mineral phases

#######################################################################
### --------------------- Aitchison Distance -----------------------###
#######################################################################

phase_names = df_PM_rolling_wt_per.columns[4:-1]
column_names = ['i', 'j', 'k', 'Run Aitch6', 'Run SigmaAll','MNR Vol%','FNR Vol%','QGAB Vol%', 'GRAN Vol%', 'MNR Aitch6',
                'FNR Aitch6','QGAB Aitch6','GRAN Aitch6', 'Run Aitch5', 'Run Aitch4', 'Run Aitch3', 'Run Sigma6', 
                'MNR SiO2' , 'MNR TiO2' ,'MNR Al2O3' ,'MNR Fe2O3' ,'MNR Cr2O3' ,'MNR FeO' ,'MNR MnO' ,'MNR MgO' ,'MNR NiO' ,'MNR CoO' ,'MNR CaO' ,'MNR Na2O' ,'MNR K2O' , 'MNR P2O5' ,
                'FNR SiO2' , 'FNR TiO2' ,'FNR Al2O3' ,'FNR Fe2O3' ,'FNR Cr2O3' ,'FNR FeO' ,'FNR MnO' ,'FNR MgO' ,'FNR NiO' ,'FNR CoO' ,'FNR CaO' ,'FNR Na2O' ,'FNR K2O' , 'FNR P2O5' ,
                'QGAB SiO2', 'QGAB TiO2','QGAB Al2O3','QGAB Fe2O3','QGAB Cr2O3','QGAB FeO','QGAB MnO','QGAB MgO','QGAB NiO','QGAB CoO','QGAB CaO','QGAB Na2O','QGAB K2O', 'QGAB P2O5',
                'GRAN SiO2', 'GRAN TiO2','GRAN Al2O3','GRAN Fe2O3','GRAN Cr2O3','GRAN FeO','GRAN MnO','GRAN MgO','GRAN NiO','GRAN CoO','GRAN CaO','GRAN Na2O','GRAN K2O', 'GRAN P2O5']
set_length = len(column_names)

# Which phases appear in a MELTS simulation can change, this section 
dynamic_columns = []
layers = ['MNR', 'FNR', 'QGAB']
for layer in layers:
    for phase in phase_names:
        dynamic_columns.append(f"{layer} {phase}")

column_names = column_names + dynamic_columns
aitch_log = np.zeros((1, len(column_names)))

#######################################################################
### --------------------- Mafic Norite (MNR) -----------------------###
#######################################################################

df_MNR = pd.DataFrame()
aitch_pos =[2, 4, 7, 9, 12, 13] # This is the position where SiO2, Al2O3, FeO, CaO, MgO and Na2O are in the df_rolling_wt_per dataframe

# this is the convoluted equation of the Aitchison distance it takes the form of:
# sum_i(1 to 6)[clr(x_i) - clr(y_i)]^2
# where clr(x_i) = log(x_i/g(x)) and clr(y_i) = log(y_i/g(y))
# g(x), g(y) is the geometric mean (x_1 * x_2 * ... * x_n)**(1/n)
# x is the weight percentage from MELTS simulation results, y is values from observation

df_MNR['Geo_x'] = (df_rolling_wt_per.iloc[:, aitch_pos].prod(axis=1) ** (1/6))
df_MNR['Aitch6'] = np.sqrt(
                  (np.log10((df_rolling_wt_per['SiO2'])  / df_MNR['Geo_x']) - np.log10(dry_MNR[0] / (dry_MNR[0]*dry_MNR[2]*dry_MNR[5]*dry_MNR[7]*dry_MNR[10]*dry_MNR[11]) ** (1/6))) ** 2 + 
                  (np.log10((df_rolling_wt_per['Al2O3']) / df_MNR['Geo_x']) - np.log10(dry_MNR[2] / (dry_MNR[0]*dry_MNR[2]*dry_MNR[5]*dry_MNR[7]*dry_MNR[10]*dry_MNR[11]) ** (1/6))) ** 2 +
                  (np.log10((df_rolling_wt_per['FeO'])   / df_MNR['Geo_x']) - np.log10(dry_MNR[5] / (dry_MNR[0]*dry_MNR[2]*dry_MNR[5]*dry_MNR[7]*dry_MNR[10]*dry_MNR[11]) ** (1/6))) ** 2 +
                  (np.log10((df_rolling_wt_per['MgO'])   / df_MNR['Geo_x']) - np.log10(dry_MNR[7] / (dry_MNR[0]*dry_MNR[2]*dry_MNR[5]*dry_MNR[7]*dry_MNR[10]*dry_MNR[11]) ** (1/6))) ** 2 +  
                  (np.log10((df_rolling_wt_per['CaO'])   / df_MNR['Geo_x']) - np.log10(dry_MNR[10]/ (dry_MNR[0]*dry_MNR[2]*dry_MNR[5]*dry_MNR[7]*dry_MNR[10]*dry_MNR[11]) ** (1/6))) ** 2 + 
                  (np.log10((df_rolling_wt_per['Na2O'])  / df_MNR['Geo_x']) - np.log10(dry_MNR[11]/ (dry_MNR[0]*dry_MNR[2]*dry_MNR[5]*dry_MNR[7]*dry_MNR[10]*dry_MNR[11]) ** (1/6))) ** 2 )

# Aitchison 5 is the Aitchison Distance but now we ignore Na2O
df_MNR['Geo_x5'] = (df_rolling_wt_per['SiO2'] * df_rolling_wt_per['Al2O3'] * df_rolling_wt_per['FeO'] * df_rolling_wt_per['MgO'] * df_rolling_wt_per['CaO']) **(1/5)
df_MNR['Aitch5'] = np.sqrt(
                  (np.log10((df_rolling_wt_per['SiO2']) / df_MNR['Geo_x5']) - np.log10(dry_MNR[0] / (dry_MNR[0]*dry_MNR[2]*dry_MNR[5]*dry_MNR[7]*dry_MNR[10]) ** (1/5))) **2 + 
                  (np.log10((df_rolling_wt_per['Al2O3'])/ df_MNR['Geo_x5']) - np.log10(dry_MNR[2] / (dry_MNR[0]*dry_MNR[2]*dry_MNR[5]*dry_MNR[7]*dry_MNR[10]) ** (1/5))) **2 + 
                  (np.log10((df_rolling_wt_per['FeO'])  / df_MNR['Geo_x5']) - np.log10(dry_MNR[5] / (dry_MNR[0]*dry_MNR[2]*dry_MNR[5]*dry_MNR[7]*dry_MNR[10]) ** (1/5))) **2 + 
                  (np.log10((df_rolling_wt_per['MgO'])  / df_MNR['Geo_x5']) - np.log10(dry_MNR[7] / (dry_MNR[0]*dry_MNR[2]*dry_MNR[5]*dry_MNR[7]*dry_MNR[10]) ** (1/5))) **2 + 
                  (np.log10((df_rolling_wt_per['CaO'])  / df_MNR['Geo_x5']) - np.log10(dry_MNR[10]/ (dry_MNR[0]*dry_MNR[2]*dry_MNR[5]*dry_MNR[7]*dry_MNR[10]) ** (1/5))) **2 ) 

# Aitchison 4 is the Aitchison Distance but now we ignore Na2O and CaO
df_MNR['Geo_x4'] = (df_rolling_wt_per['SiO2']*df_rolling_wt_per['Al2O3']*df_rolling_wt_per['FeO']*df_rolling_wt_per['MgO']) **(1/4)
df_MNR['Aitch4'] = np.sqrt( 
                  (np.log10((df_rolling_wt_per['SiO2']) / df_MNR['Geo_x4']) - np.log10(dry_MNR[0] / (dry_MNR[0]*dry_MNR[2]*dry_MNR[5]*dry_MNR[7]) ** (1/4))) ** 2 + 
                  (np.log10((df_rolling_wt_per['Al2O3'])/ df_MNR['Geo_x4']) - np.log10(dry_MNR[2] / (dry_MNR[0]*dry_MNR[2]*dry_MNR[5]*dry_MNR[7]) ** (1/4))) ** 2 + 
                  (np.log10((df_rolling_wt_per['FeO'])  / df_MNR['Geo_x4']) - np.log10(dry_MNR[5] / (dry_MNR[0]*dry_MNR[2]*dry_MNR[5]*dry_MNR[7]) ** (1/4))) ** 2 + 
                  (np.log10((df_rolling_wt_per['MgO'])  / df_MNR['Geo_x4']) - np.log10(dry_MNR[7] / (dry_MNR[0]*dry_MNR[2]*dry_MNR[5]*dry_MNR[7]) ** (1/4))) ** 2 ) 

# Aitchison 3 is the Aitchison Distance but now we ignore Na2O, CaO and FeO
df_MNR['Geo_x3'] = (df_rolling_wt_per['SiO2']*df_rolling_wt_per['Al2O3']*df_rolling_wt_per['MgO']) **(1/3)
df_MNR['Aitch3'] = np.sqrt( 
                  (np.log10((df_rolling_wt_per['SiO2']) / df_MNR['Geo_x3']) - np.log10(dry_MNR[0] / (dry_MNR[0]*dry_MNR[2]*dry_MNR[7]) ** (1/3))) ** 2 + 
                  (np.log10((df_rolling_wt_per['Al2O3'])/ df_MNR['Geo_x3']) - np.log10(dry_MNR[2] / (dry_MNR[0]*dry_MNR[2]*dry_MNR[7]) ** (1/3))) ** 2 + 
                  (np.log10((df_rolling_wt_per['MgO'])  / df_MNR['Geo_x3']) - np.log10(dry_MNR[7] / (dry_MNR[0]*dry_MNR[2]*dry_MNR[7]) ** (1/3))) ** 2 ) 

# We can also check the values of sigma squared with all oxides
df_MNR['σ^2 all'] = (((df_rolling_wt_per.iloc[:, 2:] - (100/sigma_n))**2).sum(axis=1)) / sigma_n
# And sigma squared with the top 6 oxides
df_MNR['mu6'] = (df_rolling_wt_per.iloc[:, 2] + df_rolling_wt_per.iloc[:, 4] + df_rolling_wt_per.iloc[:, 7] + df_rolling_wt_per.iloc[:, 9] +df_rolling_wt_per.iloc[:, 12] + 
                 df_rolling_wt_per.iloc[:, 13]) / 6
df_MNR['σ^2 6'] = ((df_rolling_wt_per.iloc[:, 2] - df_MNR['mu6'])**2 + (df_rolling_wt_per.iloc[:, 4] - df_MNR['mu6'])**2 + (df_rolling_wt_per.iloc[:, 7] - df_MNR['mu6'])**2 +
                   (df_rolling_wt_per.iloc[:, 9] - df_MNR['mu6'])**2 + (df_rolling_wt_per.iloc[:, 12]- df_MNR['mu6'])**2 + (df_rolling_wt_per.iloc[:, 13]- df_MNR['mu6'])**2) / 6

# We have now calculated Aitchison Distance at every step for MNR, the first layer. However, we want to add a 2nd layer, and
# calculate its subsequent Aitchison Distance as well. But where does this second layer start? In this following section the
# code will look at the numerous permutations of "i" which is the nth row of df_MNR for determining where to start the next
# layer. The 2nd layer cannot start on the first row (need a non-zero first layer) and cannot end on the last 2 rows (there
# are still 2 more layers to permutate through.

count = 0
liquid_start = df_rolling_wt_per['index'][0] - 1 # this is the LAST index in the MELTS output files before solids begin to form

for i in i_values:
    rolling_volume_fnr = pd.DataFrame()
    rolling_volume_fnr = rolling_volume.iloc[(i+1):,[0, -2, -1]].copy()
    rolling_volume_fnr['Vol %'] = rolling_volume_fnr['Vol %'] - rolling_volume['Vol %'].iloc[i]   
    vol_percent_filtered_fnr = rolling_volume_fnr[(rolling_volume_fnr['Vol %'] > bounds['FNR']['lower']) 
                                                & (rolling_volume_fnr['Vol %'] < bounds['FNR']['upper'])]

    j_values = vol_percent_filtered_fnr['index'].to_numpy() - rolling_volume_fnr['index'].iloc[0]

#######################################################################
### --------------------- Felsic Norite (FNR) ----------------------###
#######################################################################

    df_rolling_mass_fnr = pd.DataFrame()
    df_rolling_mass_fnr['index'] = df_mass_by_step.iloc[(i+1):]['index']
    df_rolling_mass_fnr[df_mass_by_step.columns[1:]] = df_mass_by_step.iloc[(i+1):, 1:].cumsum()
    df_rolling_wt_per_fnr = df_rolling_mass_fnr.copy()
    df_rolling_wt_per_fnr.iloc[:, 2:] = df_rolling_wt_per_fnr.iloc[:, 2:] / df_rolling_wt_per_fnr.iloc[:, [1]].values * 100 
    df_FNR = pd.DataFrame()
    
    # In this section we calculate the Phases masses
    df_PM_rolling_mass_fnr = pd.DataFrame()
    df_PM_rolling_mass_fnr['index'] = df_solids.iloc[(i+1):]['index']
    df_PM_rolling_mass_fnr[df_solids.columns[4:]] = df_solids.iloc[(i+1):, 4:].cumsum()
    df_PM_rolling_wt_per_fnr = df_PM_rolling_mass_fnr.copy()
    df_PM_rolling_wt_per_fnr.iloc[:, 4:-1] = df_PM_rolling_wt_per_fnr.iloc[:, 4:-1] / df_PM_rolling_wt_per_fnr.iloc[:, [-1]].values * 100

    df_PM_step_wt_per_fnr = df_solids.iloc[(i+1):, 4:]
    df_PM_step_wt_per_fnr.iloc[:, 4:-1] = df_PM_step_wt_per_fnr.iloc[:, 4:-1] / df_PM_step_wt_per_fnr.iloc[:, [-1]].values * 100

    Geo_y = (dry_FNR[0]*dry_FNR[2]*dry_FNR[5]*dry_FNR[7]*dry_FNR[10]*dry_FNR[11]) ** (1/6)
    df_FNR['Geo_x']  = (df_rolling_wt_per_fnr.iloc[:, aitch_pos].prod(axis=1) ** (1/6))
    df_FNR['Aitch6'] = np.sqrt(
                      (np.log10(df_rolling_wt_per_fnr['SiO2'] / df_FNR['Geo_x']) - (np.log10(dry_FNR[0]  / Geo_y)))**2 + 
                      (np.log10(df_rolling_wt_per_fnr['Al2O3']/ df_FNR['Geo_x']) - (np.log10(dry_FNR[2]  / Geo_y)))**2 +
                      (np.log10(df_rolling_wt_per_fnr['FeO']  / df_FNR['Geo_x']) - (np.log10(dry_FNR[5]  / Geo_y)))**2 +
                      (np.log10(df_rolling_wt_per_fnr['MgO']  / df_FNR['Geo_x']) - (np.log10(dry_FNR[7]  / Geo_y)))**2 +
                      (np.log10(df_rolling_wt_per_fnr['CaO']  / df_FNR['Geo_x']) - (np.log10(dry_FNR[10] / Geo_y)))**2 +
                      (np.log10(df_rolling_wt_per_fnr['Na2O'] / df_FNR['Geo_x']) - (np.log10(dry_FNR[11] / Geo_y)))**2 ) 

    df_FNR['Geo_x5'] = (df_rolling_wt_per_fnr['SiO2']*df_rolling_wt_per_fnr['Al2O3']*df_rolling_wt_per_fnr['FeO']*df_rolling_wt_per_fnr['MgO']*df_rolling_wt_per_fnr['CaO']) ** (1/5)
    Geo_y5 = (dry_FNR[0]*dry_FNR[2]*dry_FNR[5]*dry_FNR[7]*dry_FNR[10]) ** (1/5)
    df_FNR['Aitch5'] = np.sqrt(
                      (np.log10(df_rolling_wt_per['SiO2'] / df_FNR['Geo_x']) - np.log10(dry_FNR[0] / Geo_y5))**2 + 
                      (np.log10(df_rolling_wt_per['Al2O3']/ df_FNR['Geo_x']) - np.log10(dry_FNR[2] / Geo_y5))**2 + 
                      (np.log10(df_rolling_wt_per['FeO']  / df_FNR['Geo_x']) - np.log10(dry_FNR[5] / Geo_y5))**2 + 
                      (np.log10(df_rolling_wt_per['MgO']  / df_FNR['Geo_x']) - np.log10(dry_FNR[7] / Geo_y5))**2 + 
                      (np.log10(df_rolling_wt_per['CaO']  / df_FNR['Geo_x']) - np.log10(dry_FNR[10]/ Geo_y5))**2 )

    df_FNR['Geo_x4'] = (df_rolling_wt_per_fnr['SiO2'] * df_rolling_wt_per_fnr['Al2O3'] * df_rolling_wt_per_fnr['FeO'] * df_rolling_wt_per_fnr['MgO'] ) ** (1/4)
    Geo_y4 = (dry_FNR[0] * dry_FNR[2] * dry_FNR[5] * dry_FNR[7]) ** (1/4)
    df_FNR['Aitch4'] = np.sqrt(
                      (np.log10(df_rolling_wt_per_fnr['SiO2'] / df_FNR['Geo_x4']) - np.log10(dry_FNR[0] / Geo_y4))**2 +
                      (np.log10(df_rolling_wt_per_fnr['Al2O3']/ df_FNR['Geo_x4']) - np.log10(dry_FNR[2] / Geo_y4))**2 +
                      (np.log10(df_rolling_wt_per_fnr['FeO']  / df_FNR['Geo_x4']) - np.log10(dry_FNR[5] / Geo_y4))**2 +
                      (np.log10(df_rolling_wt_per_fnr['MgO']  / df_FNR['Geo_x4']) - np.log10(dry_FNR[7] / Geo_y4))**2 )

    df_FNR['Geo_x3'] = (df_rolling_wt_per_fnr['SiO2'] * df_rolling_wt_per_fnr['Al2O3'] * df_rolling_wt_per_fnr['MgO'] ) ** (1/3)
    Geo_y3 = (dry_FNR[0] * dry_FNR[2] * dry_FNR[7]) ** (1/3)
    df_FNR['Aitch3'] = np.sqrt(
                      (np.log10(df_rolling_wt_per_fnr['SiO2'] / df_FNR['Geo_x3']) - np.log10(dry_FNR[0] / Geo_y3))**2 +
                      (np.log10(df_rolling_wt_per_fnr['Al2O3']/ df_FNR['Geo_x3']) - np.log10(dry_FNR[2] / Geo_y3))**2 +
                      (np.log10(df_rolling_wt_per_fnr['MgO']  / df_FNR['Geo_x3']) - np.log10(dry_FNR[7] / Geo_y3))**2 )

    df_FNR['σ^2 all'] = (((df_rolling_wt_per_fnr.iloc[:, 2:15] - (100/13))**2).sum(axis=1)) / sigma_n
    df_FNR['mu6'] = (df_rolling_wt_per_fnr.iloc[:, 2] + df_rolling_wt_per_fnr.iloc[:, 4] + df_rolling_wt_per_fnr.iloc[:, 7] +
                     df_rolling_wt_per_fnr.iloc[:, 9] + df_rolling_wt_per_fnr.iloc[:, 12] + df_rolling_wt_per_fnr.iloc[:, 13]) / 6

    df_FNR['σ^2 6'] = ((df_rolling_wt_per_fnr.iloc[:, 2]  - df_FNR['mu6'])**2 + (df_rolling_wt_per_fnr.iloc[:, 4]  - df_FNR['mu6'])**2 +
                       (df_rolling_wt_per_fnr.iloc[:, 7]  - df_FNR['mu6'])**2 + (df_rolling_wt_per_fnr.iloc[:, 9]  - df_FNR['mu6'])**2 + 
                       (df_rolling_wt_per_fnr.iloc[:, 12] - df_FNR['mu6'])**2 + (df_rolling_wt_per_fnr.iloc[:, 13] - df_FNR['mu6'])**2) / 6

# We have now calculated Aitchison Distance at every step for MNR and FNR, the two layers. However, we want to add a 3rd layer, and 
# calculate its subsequent Aitchison Distance as well just like in the last step. In this following section the code will look at the
# numerous permutations of "j" which is the nth row of df_FNR for determining where to start the next layer and is the (j+i)th row of
# df_MNR. The 3rd layer cannot start on the 1st or 2nd row (need a non-zero first two layers) and cannot end on the last row since there
# is still 1 more layer to permutate through. 

    for j in j_values:

        rolling_volume_qgab = rolling_volume_fnr.iloc[(j+1):,:].copy()
        rolling_volume_qgab.loc[:, 'Vol %'] = rolling_volume_qgab['Vol %'] - rolling_volume_fnr['Vol %'].iloc[j]
        vol_percent_filtered_qgab = rolling_volume_qgab[(rolling_volume_qgab['Vol %'] > bounds['QGAB']['lower']) 
                                                      & (rolling_volume_qgab['Vol %'] < bounds['QGAB']['upper'])]
                                                      #& (rolling_volume_qgab['Remain Vol %'] < bounds['QGAB']['upper'])
                                                      #& (rolling_volume_qgab['Remain Vol %'] > bounds['QGAB']['lower'])]

        k_values = vol_percent_filtered_qgab['index'].to_numpy() - rolling_volume_qgab['index'].iloc[0]
        df_rolling_mass_qgab = pd.DataFrame()
        df_rolling_mass_qgab['index'] = df_mass_by_step.iloc[(j+i+2):]['index']
        df_rolling_mass_qgab[df_mass_by_step.columns[1:]] = df_mass_by_step.iloc[(j+i+2):, 1:].cumsum()
        df_rolling_wt_per_qgab = df_rolling_mass_qgab.copy()
        df_rolling_wt_per_qgab.iloc[:, 2:] = df_rolling_wt_per_qgab.iloc[:, 2:] / df_rolling_wt_per_qgab.iloc[:, [1]].values * 100

        # In this section we calculate the Phases masses
        df_PM_rolling_mass_qgab = pd.DataFrame()
        df_PM_rolling_mass_qgab['index'] = df_solids.iloc[(j+i+2):]['index']
        df_PM_rolling_mass_qgab[df_solids.columns[4:]] = df_solids.iloc[(j+i+2):, 4:].cumsum()
        df_PM_rolling_wt_per_qgab = df_PM_rolling_mass_qgab.copy()
        df_PM_rolling_wt_per_qgab.iloc[:, 1:-1] = df_PM_rolling_wt_per_qgab.iloc[:, 1:-1] / df_PM_rolling_wt_per_qgab.iloc[:, [-1]].values * 100

        df_PM_step_wt_per_qgab = df_solids.iloc[(j+i+2):, 4:]
        df_PM_step_wt_per_qgab.iloc[:, 4:-1] = df_PM_step_wt_per_qgab.iloc[:, 4:-1] / df_PM_step_wt_per_qgab.iloc[:, [-1]].values * 100

        df_QGAB = pd.DataFrame()
        Geo_y = (dry_QGAB[0] * dry_QGAB[2] * dry_QGAB[5] * dry_QGAB[7] * dry_QGAB[10] * dry_QGAB[11])**(1/6)
        df_QGAB['Geo_x'] = (df_rolling_wt_per_qgab.iloc[:, aitch_pos].prod(axis=1) ** (1/6))
        df_QGAB['Aitch6'] = np.sqrt(
                           (np.log10(df_rolling_wt_per_qgab['SiO2'] / df_QGAB['Geo_x']) - np.log10(dry_QGAB[0] / Geo_y))**2 +
                           (np.log10(df_rolling_wt_per_qgab['Al2O3']/ df_QGAB['Geo_x']) - np.log10(dry_QGAB[2] / Geo_y))**2 +
                           (np.log10(df_rolling_wt_per_qgab['FeO']  / df_QGAB['Geo_x']) - np.log10(dry_QGAB[5] / Geo_y))**2 +
                           (np.log10(df_rolling_wt_per_qgab['MgO']  / df_QGAB['Geo_x']) - np.log10(dry_QGAB[7] / Geo_y))**2 +
                           (np.log10(df_rolling_wt_per_qgab['CaO']  / df_QGAB['Geo_x']) - np.log10(dry_QGAB[10]/ Geo_y))**2 +
                           (np.log10(df_rolling_wt_per_qgab['Na2O'] / df_QGAB['Geo_x']) - np.log10(dry_QGAB[11]/ Geo_y))**2 )

        Geo_y = (dry_QGAB[0] * dry_QGAB[2] * dry_QGAB[5] * dry_QGAB[7] * dry_QGAB[10] )**(1/5)
        df_QGAB['Geo_x'] = (df_rolling_wt_per_qgab['SiO2'] * df_rolling_wt_per_qgab['Al2O3'] * df_rolling_wt_per_qgab['FeO'] * df_rolling_wt_per_qgab['MgO'] * df_rolling_wt_per_qgab['CaO'] ) ** (1/5)
        df_QGAB['Aitch5'] = np.sqrt(
                           (np.log10(df_rolling_wt_per_qgab['SiO2'] / df_QGAB['Geo_x']) - np.log10(dry_QGAB[0] / Geo_y))**2 +
                           (np.log10(df_rolling_wt_per_qgab['Al2O3']/ df_QGAB['Geo_x']) - np.log10(dry_QGAB[2] / Geo_y))**2 +
                           (np.log10(df_rolling_wt_per_qgab['FeO']  / df_QGAB['Geo_x']) - np.log10(dry_QGAB[5] / Geo_y))**2 +
                           (np.log10(df_rolling_wt_per_qgab['MgO']  / df_QGAB['Geo_x']) - np.log10(dry_QGAB[7] / Geo_y))**2 +
                           (np.log10(df_rolling_wt_per_qgab['CaO']  / df_QGAB['Geo_x']) - np.log10(dry_QGAB[10]/ Geo_y))**2 )
 
        df_QGAB['Geo_x4'] = (df_rolling_wt_per_qgab['SiO2'] * df_rolling_wt_per_qgab['Al2O3'] * df_rolling_wt_per_qgab['FeO'] * df_rolling_wt_per_qgab['MgO'] ) ** (1/4)
        Geo_y4 = (dry_QGAB[0] * dry_QGAB[2] * dry_QGAB[5] * dry_QGAB[7]) ** (1/4)
        df_QGAB['Aitch4'] = np.sqrt(
                           (np.log10(df_rolling_wt_per_qgab['SiO2'] / df_QGAB['Geo_x4']) - np.log10(dry_QGAB[0] / Geo_y4))**2 +
                           (np.log10(df_rolling_wt_per_qgab['Al2O3']/ df_QGAB['Geo_x4']) - np.log10(dry_QGAB[2] / Geo_y4))**2 +
                           (np.log10(df_rolling_wt_per_qgab['FeO']  / df_QGAB['Geo_x4']) - np.log10(dry_QGAB[5] / Geo_y4))**2 +
                           (np.log10(df_rolling_wt_per_qgab['MgO']  / df_QGAB['Geo_x4']) - np.log10(dry_QGAB[7] / Geo_y4))**2 )

        df_QGAB['Geo_x3'] = (df_rolling_wt_per_qgab['SiO2'] * df_rolling_wt_per_qgab['Al2O3'] * df_rolling_wt_per_qgab['MgO'] ) ** (1/3)
        Geo_y3 = (dry_QGAB[0] * dry_QGAB[2] * dry_QGAB[7]) ** (1/3)
        df_QGAB['Aitch3'] = np.sqrt(
                           (np.log10(df_rolling_wt_per_qgab['SiO2'] / df_QGAB['Geo_x3']) - np.log10(dry_QGAB[0] / Geo_y3))**2 +
                           (np.log10(df_rolling_wt_per_qgab['Al2O3']/ df_QGAB['Geo_x3']) - np.log10(dry_QGAB[2] / Geo_y3))**2 +
                           (np.log10(df_rolling_wt_per_qgab['MgO']  / df_QGAB['Geo_x3']) - np.log10(dry_QGAB[7] / Geo_y3))**2 )

        df_QGAB['σ^2 all'] = (((df_rolling_wt_per_qgab.iloc[:, 2:15] - (100/13))**2).sum(axis=1)) / sigma_n
        df_QGAB['mu6'] = (df_rolling_wt_per_qgab.iloc[:, 2] + df_rolling_wt_per_qgab.iloc[:, 4] + df_rolling_wt_per_qgab.iloc[:, 7] + 
                          df_rolling_wt_per_qgab.iloc[:, 9] + df_rolling_wt_per_qgab.iloc[:, 12]+ df_rolling_wt_per_qgab.iloc[:, 13]) / 6

        df_QGAB['σ^2 6'] = ((df_rolling_wt_per_qgab.iloc[:, 2] - df_QGAB['mu6'])**2 +
                            (df_rolling_wt_per_qgab.iloc[:, 4] - df_QGAB['mu6'])**2 +
                            (df_rolling_wt_per_qgab.iloc[:, 7] - df_QGAB['mu6'])**2 +
                            (df_rolling_wt_per_qgab.iloc[:, 9] - df_QGAB['mu6'])**2 +
                            (df_rolling_wt_per_qgab.iloc[:, 12]- df_QGAB['mu6'])**2 +
                            (df_rolling_wt_per_qgab.iloc[:, 13]- df_QGAB['mu6'])**2 ) / 6
 
        for k in k_values:

            rolling_volume_gran = pd.DataFrame()
            rolling_volume_gran = rolling_volume_qgab.iloc[(k+1):,:]
             
            # vol_percent_filtered_gran = rolling_volume_gran[(rolling_volume_gran['Remain Vol %'] > 50) & (rolling_volume_gran['Remain Vol %'] < 70)]
            # The Granophyric layer represents all of the rest of the material, which means it is the sum 
            # of the remaining solids AND the rest of the liquid (magma). The thought process here to ignore the summed solids,
            # and instead just look at the composition of the liquid at the start of this layer, since any solids will just
            # come from the magma 
 
            gran_liq = pd.read_csv('Liquid_comp_tbl.txt', delimiter=' ', skiprows=1)
            df_GRAN = gran_liq.iloc[k + j + i + 3 + liquid_start]
            
            Geo_x = (df_GRAN['SiO2'] * df_GRAN['Al2O3'] * df_GRAN['FeO'] * df_GRAN['MgO'] * df_GRAN['CaO'] * df_GRAN['Na2O']) ** (1/6)
            Geo_y = (wet_GRAN[0] * wet_GRAN[2] * wet_GRAN[5] * wet_GRAN[7] * wet_GRAN[10] * wet_GRAN[11]) ** (1/6)
            ClrSiO2    = np.log10((df_GRAN['SiO2']) / Geo_x) - np.log10(wet_GRAN[0] / Geo_y )
            ClrAl2O3   = np.log10((df_GRAN['Al2O3'])/ Geo_x) - np.log10(wet_GRAN[2] / Geo_y )
            ClrFeO     = np.log10((df_GRAN['FeO'])  / Geo_x) - np.log10(wet_GRAN[5] / Geo_y )
            ClrMgO     = np.log10((df_GRAN['MgO'])  / Geo_x) - np.log10(wet_GRAN[7] / Geo_y )
            ClrCaO     = np.log10((df_GRAN['CaO'])  / Geo_x) - np.log10(wet_GRAN[10]/ Geo_y )
            ClrNa2O    = np.log10((df_GRAN['Na2O']) / Geo_x) - np.log10(wet_GRAN[11]/ Geo_y )
            gran_aitch = np.round((ClrSiO2**2 + ClrAl2O3**2 + ClrFeO**2 + ClrMgO **2 + ClrCaO**2 + ClrNa2O**2)** 0.5, 3)
            
            Geo_x_5 = (df_GRAN['SiO2']*df_GRAN['Al2O3']*df_GRAN['FeO']*df_GRAN['MgO']*df_GRAN['CaO']) ** (1/5)
            Geo_y_5 = (wet_GRAN[0]*wet_GRAN[2]*wet_GRAN[5]*wet_GRAN[7]*wet_GRAN[10]) ** (1/5)
            ClrSiO2_5    = np.log10((df_GRAN['SiO2']) / Geo_x_5) - np.log10(wet_GRAN[0] / Geo_y_5 )
            ClrAl2O3_5   = np.log10((df_GRAN['Al2O3'])/ Geo_x_5) - np.log10(wet_GRAN[2] / Geo_y_5 )
            ClrFeO_5     = np.log10((df_GRAN['FeO'])  / Geo_x_5) - np.log10(wet_GRAN[5] / Geo_y_5 )
            ClrMgO_5     = np.log10((df_GRAN['MgO'])  / Geo_x_5) - np.log10(wet_GRAN[7] / Geo_y_5 )
            ClrCaO_5     = np.log10((df_GRAN['CaO'])  / Geo_x_5) - np.log10(wet_GRAN[10]/ Geo_y_5 )
            gran_aitch_5 = np.round((ClrSiO2_5**2 + ClrAl2O3_5**2 + ClrFeO_5**2 + ClrMgO_5 **2 + ClrCaO_5**2 )** 0.5, 3)

            Geo_x_4 = (df_GRAN['SiO2']*df_GRAN['Al2O3']*df_GRAN['FeO']*df_GRAN['MgO']) ** (1/4)
            Geo_y_4 = (wet_GRAN[0]*wet_GRAN[2]*wet_GRAN[5]*wet_GRAN[7]) ** (1/4)
            ClrSiO2_4  = np.log10((df_GRAN['SiO2']) / Geo_x_4) - np.log10(wet_GRAN[0] / Geo_y_4 )
            ClrAl2O3_4 = np.log10((df_GRAN['Al2O3'])/ Geo_x_4) - np.log10(wet_GRAN[2] / Geo_y_4 )
            ClrFeO_4   = np.log10((df_GRAN['FeO'])  / Geo_x_4) - np.log10(wet_GRAN[5] / Geo_y_4 )
            ClrMgO_4   = np.log10((df_GRAN['MgO'])  / Geo_x_4) - np.log10(wet_GRAN[7] / Geo_y_4 )
            gran_aitch_4 = np.round((ClrSiO2_4**2 + ClrAl2O3_4**2 + ClrFeO_4**2 + ClrMgO_4 **2 )** 0.5, 3)

            Geo_x_3 = (df_GRAN['SiO2']*df_GRAN['Al2O3']*df_GRAN['MgO']) ** (1/3)
            Geo_y_3 = (wet_GRAN[0]*wet_GRAN[2]*wet_GRAN[7]) ** (1/3)
            ClrSiO2_3    = np.log10((df_GRAN['SiO2']) / Geo_x_3) - np.log10(wet_GRAN[0] / Geo_y_3 )
            ClrAl2O3_3   = np.log10((df_GRAN['Al2O3'])/ Geo_x_3) - np.log10(wet_GRAN[2] / Geo_y_3 )
            ClrFeO_3     = np.log10((df_GRAN['FeO'])  / Geo_x_3) - np.log10(wet_GRAN[5] / Geo_y_3 )
            ClrMgO_3     = np.log10((df_GRAN['MgO'])  / Geo_x_3) - np.log10(wet_GRAN[7] / Geo_y_3 )
            ClrCaO_3     = np.log10((df_GRAN['CaO'])  / Geo_x_3) - np.log10(wet_GRAN[10]/ Geo_y_3 )
            gran_aitch_3 = np.round((ClrSiO2_3**2 + ClrAl2O3_3**2 + ClrMgO_3 **2 )** 0.5, 3)
            
            gran_sigma_all = ((df_GRAN['SiO2']  - (100/sigma_n))**2 + (df_GRAN['TiO2']  - (100/sigma_n))**2 + (df_GRAN['Al2O3'] - (100/sigma_n))**2 +
                              (df_GRAN['Fe2O3'] - (100/sigma_n))**2 + (df_GRAN['Cr2O3'] - (100/sigma_n))**2 + (df_GRAN['FeO']   - (100/sigma_n))**2 +
                              (df_GRAN['MnO']   - (100/sigma_n))**2 + (df_GRAN['MgO']   - (100/sigma_n))**2 + (df_GRAN['NiO']   - (100/sigma_n))**2 +
                              (df_GRAN['CoO']   - (100/sigma_n))**2 + (df_GRAN['CaO']   - (100/sigma_n))**2 + (df_GRAN['Na2O']  - (100/sigma_n))**2 +
                              (df_GRAN['K2O']   - (100/sigma_n))**2 + (df_GRAN['P2O5']  - (100/sigma_n))**2 + (df_GRAN['H2O']   - (100/sigma_n))**2 +
                              (df_GRAN['CO2']   - (100/sigma_n))**2)/ 16
            
            gran_mu = (df_GRAN['SiO2'] + df_GRAN['Al2O3'] + df_GRAN['FeO'] + df_GRAN['MgO'] + df_GRAN['CaO'] + df_GRAN['Na2O'])/6
            gran_sigma_6 = ((df_GRAN['SiO2'] - (gran_mu))**2 + (df_GRAN['Al2O3'] - (gran_mu))**2 + (df_GRAN['FeO'] - (gran_mu))**2  + 
                            (df_GRAN['MgO'] - (gran_mu))**2 + (df_GRAN['CaO'] - (gran_mu))**2 + (df_GRAN['Na2O'] - (gran_mu))**2 )/ 6

#######################################################################
### ------------------------ Data Analysis -------------------------###
#######################################################################
            mnr_vol_frac  = np.round(rolling_volume.iloc[i]['Vol %']     , 3)
            fnr_vol_frac  = np.round(rolling_volume_fnr.iloc[j]['Vol %'] , 3)
            qgab_vol_frac = np.round(rolling_volume_qgab.iloc[k]['Vol %'], 3)
            gran_vol_frac = np.round(rolling_volume_qgab.iloc[k]['Remain Vol %'], 3) # 100 - (mnr_vol_frac + fnr_vol_frac + qgab_vol_frac)
            
            mnr_aitch  = np.round(df_MNR.iloc[i]['Aitch6'] , 3)
            fnr_aitch  = np.round(df_FNR.iloc[j]['Aitch6'] , 3)
            qgab_aitch = np.round(df_QGAB.iloc[k]['Aitch6'], 3)
            running_aitch = np.round(mnr_aitch + fnr_aitch + qgab_aitch + gran_aitch, 3)

            aitch_log[count,0] = i
            aitch_log[count,1] = j
            aitch_log[count,2] = k
            aitch_log[count,3] = running_aitch
            aitch_log[count,4] = np.round(df_MNR.iloc[i]['σ^2 all'] + df_FNR.iloc[j]['σ^2 all'] + df_QGAB.iloc[k]['σ^2 all'] + gran_sigma_all, 3)

            aitch_log[count,5] = mnr_vol_frac
            aitch_log[count,6] = fnr_vol_frac
            aitch_log[count,7] = qgab_vol_frac
            aitch_log[count,8] = gran_vol_frac

            aitch_log[count,9]  = np.round(df_MNR.iloc[i]['Aitch6'] , 3)
            aitch_log[count,10] = np.round(df_FNR.iloc[j]['Aitch6'] , 3)
            aitch_log[count,11] = np.round(df_QGAB.iloc[k]['Aitch6'], 3)
            aitch_log[count,12] = gran_aitch

            aitch_log[count,13] = np.round(df_MNR.iloc[i]['Aitch5'] + df_FNR.iloc[j]['Aitch5'] + df_QGAB.iloc[k]['Aitch5'] + gran_aitch_5, 3)
            aitch_log[count,14] = np.round(df_MNR.iloc[i]['Aitch4'] + df_FNR.iloc[j]['Aitch4'] + df_QGAB.iloc[k]['Aitch4'] + gran_aitch_4, 3)
            aitch_log[count,15] = np.round(df_MNR.iloc[i]['Aitch3'] + df_FNR.iloc[j]['Aitch3'] + df_QGAB.iloc[k]['Aitch3'] + gran_aitch_3, 3)
            aitch_log[count,16] = np.round(df_MNR.iloc[i]['σ^2 6']  + df_FNR.iloc[j]['σ^2 6']  + df_QGAB.iloc[k]['σ^2 6']  + gran_sigma_6, 3)

            aitch_log[count,17] = np.round(df_rolling_wt_per.iloc[i]['SiO2'],  2)
            aitch_log[count,18] = np.round(df_rolling_wt_per.iloc[i]['TiO2'],  2)
            aitch_log[count,19] = np.round(df_rolling_wt_per.iloc[i]['Al2O3'], 2)
            aitch_log[count,20] = np.round(df_rolling_wt_per.iloc[i]['Fe2O3'], 2)
            aitch_log[count,21] = np.round(df_rolling_wt_per.iloc[i]['Cr2O3'], 2)
            aitch_log[count,22] = np.round(df_rolling_wt_per.iloc[i]['FeO'],   2)
            aitch_log[count,23] = np.round(df_rolling_wt_per.iloc[i]['MnO'],   2)
            aitch_log[count,24] = np.round(df_rolling_wt_per.iloc[i]['MgO'],   2)
            aitch_log[count,25] = np.round(df_rolling_wt_per.iloc[i]['NiO'],   2)
            aitch_log[count,26] = np.round(df_rolling_wt_per.iloc[i]['CoO'],   2)
            aitch_log[count,27] = np.round(df_rolling_wt_per.iloc[i]['CaO'],   2)
            aitch_log[count,28] = np.round(df_rolling_wt_per.iloc[i]['Na2O'],  2)
            aitch_log[count,29] = np.round(df_rolling_wt_per.iloc[i]['K2O'],   2)
            aitch_log[count,30] = np.round(df_rolling_wt_per.iloc[i]['P2O5'],  2)

            aitch_log[count,31] = np.round(df_rolling_wt_per_fnr.iloc[j]['SiO2'],  2)
            aitch_log[count,32] = np.round(df_rolling_wt_per_fnr.iloc[j]['TiO2'],  2)
            aitch_log[count,33] = np.round(df_rolling_wt_per_fnr.iloc[j]['Al2O3'], 2)
            aitch_log[count,34] = np.round(df_rolling_wt_per_fnr.iloc[j]['Fe2O3'], 2)
            aitch_log[count,35] = np.round(df_rolling_wt_per_fnr.iloc[j]['Cr2O3'], 2)
            aitch_log[count,36] = np.round(df_rolling_wt_per_fnr.iloc[j]['FeO'],   2)
            aitch_log[count,37] = np.round(df_rolling_wt_per_fnr.iloc[j]['MnO'],   2)
            aitch_log[count,38] = np.round(df_rolling_wt_per_fnr.iloc[j]['MgO'],   2)
            aitch_log[count,39] = np.round(df_rolling_wt_per_fnr.iloc[j]['NiO'],   2)
            aitch_log[count,40] = np.round(df_rolling_wt_per_fnr.iloc[j]['CoO'],   2)
            aitch_log[count,41] = np.round(df_rolling_wt_per_fnr.iloc[j]['CaO'],   2)
            aitch_log[count,42] = np.round(df_rolling_wt_per_fnr.iloc[j]['Na2O'],  2)
            aitch_log[count,43] = np.round(df_rolling_wt_per_fnr.iloc[j]['K2O'],   2)
            aitch_log[count,44] = np.round(df_rolling_wt_per_fnr.iloc[j]['P2O5'],  2)

            aitch_log[count,45] = np.round(df_rolling_wt_per_qgab.iloc[k]['SiO2'],  2)
            aitch_log[count,46] = np.round(df_rolling_wt_per_qgab.iloc[k]['TiO2'],  2)
            aitch_log[count,47] = np.round(df_rolling_wt_per_qgab.iloc[k]['Al2O3'], 2)
            aitch_log[count,48] = np.round(df_rolling_wt_per_qgab.iloc[k]['Fe2O3'], 2)
            aitch_log[count,49] = np.round(df_rolling_wt_per_qgab.iloc[k]['Cr2O3'], 2)
            aitch_log[count,50] = np.round(df_rolling_wt_per_qgab.iloc[k]['FeO'],   2)
            aitch_log[count,51] = np.round(df_rolling_wt_per_qgab.iloc[k]['MnO'],   2)
            aitch_log[count,52] = np.round(df_rolling_wt_per_qgab.iloc[k]['MgO'],   2)
            aitch_log[count,53] = np.round(df_rolling_wt_per_qgab.iloc[k]['NiO'],   2)
            aitch_log[count,54] = np.round(df_rolling_wt_per_qgab.iloc[k]['CoO'],   2)
            aitch_log[count,55] = np.round(df_rolling_wt_per_qgab.iloc[k]['CaO'],   2)
            aitch_log[count,56] = np.round(df_rolling_wt_per_qgab.iloc[k]['Na2O'],  2)
            aitch_log[count,57] = np.round(df_rolling_wt_per_qgab.iloc[k]['K2O'],   2)
            aitch_log[count,58] = np.round(df_rolling_wt_per_qgab.iloc[k]['P2O5'],  2)

            aitch_log[count,59] = np.round(df_GRAN['SiO2'],  2)
            aitch_log[count,60] = np.round(df_GRAN['TiO2'],  2)
            aitch_log[count,60] = np.round(df_GRAN['Al2O3'], 2)
            aitch_log[count,62] = np.round(df_GRAN['Fe2O3'], 2)
            aitch_log[count,63] = np.round(df_GRAN['Cr2O3'], 2)
            aitch_log[count,64] = np.round(df_GRAN['FeO'],   2)
            aitch_log[count,65] = np.round(df_GRAN['MnO'],   2)
            aitch_log[count,66] = np.round(df_GRAN['MgO'],   2)
            aitch_log[count,67] = np.round(df_GRAN['NiO'],   2)
            aitch_log[count,68] = np.round(df_GRAN['CoO'],   2)
            aitch_log[count,69] = np.round(df_GRAN['CaO'],   2)
            aitch_log[count,70] = np.round(df_GRAN['Na2O'],  2)
            aitch_log[count,71] = np.round(df_GRAN['K2O'],   2)
            aitch_log[count,72] = np.round(df_GRAN['P2O5'],  2)

            for m, phase in enumerate(phase_names, start= set_length):
                aitch_log[count, m] = df_PM_rolling_wt_per.iloc[i][phase]
            for n, phase in enumerate(phase_names, start= set_length +len(phase_names)):
                aitch_log[count, n] = df_PM_rolling_wt_per_fnr.iloc[j][phase]
            for p, phase in enumerate(phase_names, start= set_length +(2*len(phase_names))):
                aitch_log[count, p] = df_PM_rolling_wt_per_qgab.iloc[k][phase]

            new_row = np.array([[np.nan] * len(column_names)])
            aitch_log = np.vstack((aitch_log, new_row))
            count += 1

aitch_log = aitch_log[0:-1,:] # get rid of the NaN last row
end_time = time.time()
Total_time = np.round(end_time - start_time, 3)
second_rate = np.round(count/Total_time, 3)
minute_rate = np.round(second_rate * 60, 3)
hour_rate = np.round(minute_rate * 60/1000000, 3)
print('Time elapsed:', Total_time, 's. The rate is', second_rate, 'counts per second,', minute_rate,
      'counts per minute,', hour_rate, 'million counts per hour')
print('Number of Counts', count)

aitch_log_df = pd.DataFrame(aitch_log, columns=column_names)
aitch_log_df = aitch_log_df.sort_values(by='Run Aitch6', ascending=True)
np.savetxt('AitchisonValues.txt', aitch_log_df, fmt='%.6f', delimiter='\t', header='\t'.join(column_names), comments='')
