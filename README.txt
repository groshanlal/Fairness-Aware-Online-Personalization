Run as:
python simulation.py

The simulation runs fairness experiments on a online perceptron and tests the idea of using a particular kind of regularisation to achieve fairness at the cost of precision. It explores trade-offs in fairness with different choices of hyper-parameters like learning rate and number of training rounds.

model.py contains details about how data is generated.
metrics.py contains details about fairness metrics.
online_perceptron.py contains details about how the perceptron is trained for warm start and online.
simulation.py has details about the experiments run on the online perceptron
simulation_global_vars.py has details about the parameters used in running the simulations.

Link to the paper: 
https://arxiv.org/abs/2007.15270