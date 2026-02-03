import numpy as np
import matplotlib.pyplot as plt
import pySALEPlot as psp

Mass_P = 5.97e24  # kg
Radius = 6378  # km
# ===========================================================
#                    asteroid.imp Parameters
# ===========================================================
ast_input = open('asteroid.inp', 'r')
keywords = ['GRIDSPC', 'OBJRESH', 'DTSAVE', 'TR_SPCH', 'TR_SPCV', 'LAYPOS']
ast_dict = {}
mat = []
for line in ast_input:
    word = line[0:16].replace(' ', '')
    if word == 'S_TYPE':
        Type = line[54:-1].replace(' ', '')
    value = '['+(line[54:-1].replace(' ', '').replace(':', ',')).replace('D', 'e')+']'
    if word in keywords:
        ast_dict[word] = eval(value)

for i in range(len(ast_dict['LAYPOS'])):
    globals()["lay_{}".format(len(ast_dict['LAYPOS']) - i)] = ast_dict['LAYPOS'][i]

spacing = (ast_dict['GRIDSPC'][0]) * .001  # (km)
dx = (ast_dict['TR_SPCH'][0]) * -spacing  # tracer spacing horizontal
dy = (ast_dict['TR_SPCV'][0]) * -spacing  # tracer spacing vertical
imp_dia = (ast_dict['OBJRESH'][0]) * spacing * 2  # Impactor diameter
# ===========================================================
#                  Alright, Melt Finding Time
# ===========================================================
model = psp.opendatfile('Sudbury/jdata.dat')  # open the data file now
model.setScale('km')  # next we can set the distance units to be km
step = model.readStep('TrP', model.nsteps - 1)
step0 = model.readStep('TrP', 0)

# -----------All Melt------------------------
indices = np.where((step.TrP > 6e10) & (step.ymark < 300))
# vapor_indices = np.where((step.TrP > 6e10) & (step.ymark < 300))

melt_index = indices[0][:]
melt_x = np.zeros(len(melt_index))
melt_y = np.zeros(len(melt_index))
total_melt = 0

indices_deep = np.where(step.TrP > 6e10)
lowest_val_y = -np.min(step0.ymark[indices_deep])

for i in range(len(melt_index)):
    melt_x[i] = step0.xmark[melt_index[i]]
    melt_y[i] = step0.ymark[melt_index[i]]
for j in range(len(melt_x)):
    volume = 2 * np.pi * dx * dy * melt_x[j] * ((Radius + melt_y[j]) / Radius)
    total_melt += (volume)
total_melt = round(total_melt, 1)
# -----------All Melt------------------------
# ----------Impactor Melt--------------------
impactor = np.where(melt_y >= 0)
impactor_index = impactor[0][:]
impact_x = np.zeros(len(impactor_index))
impact_y = np.zeros(len(impactor_index))
impactor_melt = 0
for k in range(len(impactor_index)):  # impactor contribution
    impact_x[k] = melt_x[impactor_index[k]]
    impact_y[k] = melt_y[impactor_index[k]]
for l in range(len(impact_x)):
    volume = 2 * np.pi * dx * dy * impact_x[l] * ((Radius + impact_y[l]) / Radius)
    impactor_melt += volume
impactor_melt = round(impactor_melt, 1)
# ---------Impactor Melt---------------------
# Determine the number of layers
num_layers = len(ast_dict['LAYPOS'])

# Initialize dictionaries to store layer data
layer_melt = {}
layer_x = {}
layer_y = {}

# Loop through all layers
for i in range(num_layers):
    layer_name = "layer{}_melt".format(i + 1)
    
    if i == 0:
        # Top layer condition
        condition = (melt_y < 0) & (melt_y >= -(lay_1 - lay_2) * spacing)
    else:
        # General case for deeper layers
        condition = (melt_y <= -(lay_1 - eval("lay_{}".format(i + 1))) * spacing) & \
                    (melt_y > -(lay_1 - eval("lay_{}".format(i + 2))) * spacing) if i + 1 < num_layers else (melt_y <= -(lay_1 - eval("lay_{}".format(i + 1))) * spacing)
    
    layer_indices = np.where(condition)
    layer_x[i] = np.zeros(len(layer_indices[0]))
    layer_y[i] = np.zeros(len(layer_indices[0]))
    layer_melt[layer_name] = 0
    
    for k in range(len(layer_indices[0])):
        layer_x[i][k] = melt_x[layer_indices[0][k]]
        layer_y[i][k] = melt_y[layer_indices[0][k]]
    
    for l in range(len(layer_x[i])):
        volume = 2 * np.pi * dx * dy * layer_x[i][l] * ((Radius + layer_y[i][l]) / Radius)
        layer_melt[layer_name] += volume
    
    layer_melt[layer_name] = round(layer_melt[layer_name], 1)

# Print results
print("Total melt =", total_melt, "km^3")
print("Impactor melt =", impactor_melt, "km^3")

for i in range(num_layers):
    print("Layer {} melt =".format(i + 1), layer_melt["layer{}_melt".format(i + 1)], "km^3")

print("Deepest SIC material originates from", lowest_val_y, "km deep")
 

