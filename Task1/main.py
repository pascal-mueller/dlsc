from Pinns import Pinns
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

n_int = 256 # Interior points
n_sb = 64 # Points on each spatial boundary
n_tb = 64 # Points on temporal boundary

# 2 output dimensions
pinn = Pinns(n_int, n_sb, n_tb)

# Plot the input training points
input_sb_, output_sb_ = pinn.add_spatial_boundary_points() # Done
input_tb_, output_tb_ = pinn.add_temporal_boundary_points() # Done
input_int_, output_int_ = pinn.add_interior_points() # Done

plt.figure(figsize=(16, 8), dpi=150)
plt.scatter(input_sb_[:, 1].detach().numpy(), input_sb_[:, 0].detach().numpy(), label="Boundary Points")
plt.scatter(input_int_[:, 1].detach().numpy(), input_int_[:, 0].detach().numpy(), label="Interior Points")
plt.scatter(input_tb_[:, 1].detach().numpy(), input_tb_[:, 0].detach().numpy(), label="Initial Points")
plt.xlabel("x")
plt.ylabel("t")
plt.legend()
plt.show()

n_epochs = 1
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                              lr=float(0.5),
                              max_iter=50000,
                              max_eval=50000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=np.finfo(float).eps)

optimizer_ADAM = optim.Adam(pinn.approximate_solution.parameters(),
                            lr=float(0.001))

hist = pinn.fit(num_epochs=n_epochs+5000,
                optimizer=optimizer_LBFGS,
                verbose=True)

plt.figure(dpi=150)
plt.grid(True, which="both", ls=":")
plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
plt.xscale("log")
plt.legend()
plt.show()