# mfg_elec
Code for the MFG_ELEC model for the dynamics of electricity market

Change model parameters in files
common_params.py (parameters affecting general properties of the model, such as electricity demand or carbon price)
gas.py (parameters of gas producers)
coal.py (parameters of coal producers)
renewable.py (parameters of renewable producers)

Use the command
python simulation.py
to run the model

Change file
simulation.py
to modify model structure (types of producers etc.) and output graphs

The main model code is contained in the file
elecmarket.py
