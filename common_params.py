{
    "Nt" : 101, # Time grid size
    "tmin" : 0,
    "tmax" : 25, # simulation length in years
    "power" : 0.8, # parameters of the weight function in fictitious play
    "offset" : 5, # parameters of the weight function in fictitious play
    "iterations" : 200, # Number of iterations in fictitious play
    "tolerance" : 100, # The algorithm stops if objective improvement is less than tolerance or after the number of iterations is reached
    "carbon tax" : ([0,10,25],[30,60,200]), # Carbon tax is computed by linear interpolation: the first list contains dates and the second one values
    "demand ratio" : 1.257580634655325, # Ratio peak to offpeak demand
    "Nfuels" : 2, # Number of different fuels
    "Fsupply" : ([10, 20], [1.0, 0.5]), # Fuel supply functions; they are linear in this version of the model;
    # the first array contains the intercepts and the second one the coefficients
    "demand" : [61.23322924, 55.99534092, 55.48594115, 60.34034567, 61.0531315 ,
       55.83064873, 55.3227472 , 60.16287406, 60.38244757, 55.21733508,
       54.71501297, 59.50196982, 60.69293604, 55.50126437, 54.99635931,
       59.80793085, 60.51283831, 55.3365722 , 54.83316537, 59.63045924,
       60.33274058, 55.17188002, 54.66997143, 59.45298764, 60.15264284,
       55.00718784, 54.50677748, 59.27551603, 59.97254511, 54.84249566,
       54.34358354, 59.09804443, 59.79244737, 54.67780348, 54.18038959,
       58.92057282, 59.61234964, 54.5131113 , 54.01719565, 58.74310122,
       59.4322519 , 54.34841912, 53.8540017 , 58.56562961, 59.36021281,
       54.28254225, 53.78872412, 58.49464097, 59.28817373, 54.21666538,
       53.72344655, 58.42365234, 59.21613463, 54.15078851, 53.65816898,
       58.35266369, 59.14409554, 54.08491164, 53.5928914 , 58.28167505,
       59.07205645, 54.01903477, 53.52761382, 58.21068641, 59.00001735,
       53.9531579 , 53.46233624, 58.13969777, 58.92797826, 53.88728102,
       53.39705866, 58.06870912, 58.85593917, 53.82140415, 53.33178108,
       57.99772048, 58.78390007, 53.75552728, 53.26650351, 57.92673184,
       58.71186098, 53.68965041, 53.20122593, 57.8557432 , 58.79590659,
       53.76650676, 53.2773831 , 57.93856328, 58.87995219, 53.84336311,
       53.35354027, 58.02138336, 58.9639978 , 53.92021945, 53.42969745,
       58.10420344, 59.04804341, 53.9970758 , 53.50585462, 58.18702352,
       59.04804341]
}
