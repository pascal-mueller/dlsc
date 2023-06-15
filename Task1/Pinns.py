import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from Common import NeuralNet, MultiVariatePoly
import time

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)


class Pinns:
    def __init__(self, n_int_, n_sb_, n_tb_):
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_

        # Parameters
        self.T_hot = 4.0
        self.T_initial = 1.0
        self.alpha_fluid = 0.05
        self.alpha_solid = 0.08
        self.h_fluid = 5
        self.h_solid = 6
        self.U_fluid = 1

        # Number of space dimensions
        self.space_dimensions = 1

        # Extrema of the solution domain (t,x) in [0,1.0]x[-1,1]
        self.domain_extrema = torch.tensor(
            [[0.0, 1.0], # Time dimension
             [0.0, 1]]  # Space dimension
        )

        # Parameter to balance role of data and PDE
        self.lambda_u = 10
        
        # F Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(
            input_dimension=self.domain_extrema.shape[0],
            output_dimension=2,
            n_hidden_layers=10,
            neurons=200,
            regularization_param=0.01,
            regularization_exp=2.0,
            retrain_seed=42
        )
        """self.approximate_solution = MultiVariatePoly(self.domain_extrema.shape[0], 3)"""

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(
            dimension=self.domain_extrema.shape[0]
        )

        # Training sets S_sb, S_tb, S_int as torch dataloader
        (
            self.training_set_sb,
            self.training_set_tb,
            self.training_set_int,
        ) = self.assemble_datasets()

    ############################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert tens.shape[1] == self.domain_extrema.shape[0]
        return (
            tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0])
            + self.domain_extrema[:, 0]
        )

    # Exact solution for the heat equation ut = u_xx with the IC above
    def exact_solution(self, inputs):
        t = inputs[:, 0]
        x = inputs[:, 1]

        u = -torch.exp(-np.pi**2 * t) * torch.sin(np.pi * x)
        return u

    ############################################################################
    # Function returning the input-output tensor required to assemble the
    # training set S_tb corresponding to the temporal boundary
    def add_temporal_boundary_points(self):
        # TODO: Implement
        t0 = self.domain_extrema[0, 0] # (t,x)
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)

        # Apply initial condition
        output_tb = torch.full((input_tb.shape[0], 2), self.T_initial)

        return input_tb, output_tb

    # Function returning the input-output tensor required to assemble the
    # training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        # TODO: Implement

        # Choose left spatial boundary point
        x0 = self.domain_extrema[1, 0] # 0.0

        # Choose right spatial boundary point
        xL = self.domain_extrema[1, 1] # 1.0

        # Draw a Sobol sequence of length n_sb. This sequence is in [0, 1]
        # Convert from [0, 1] to spatial domain extrema.
        # input_sb is a 2d array (one for time, one for space???)

        # Note: I think we just get a 2d array here based on wrong dimensions
        # simply because we need 2 dimensions anyway.
        input_sb = self.convert(self.soboleng.draw(self.n_sb))

        # Left spatial boundary
        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        # Right spatial boundary
        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)
        
        # First col: time

        # We gonna build a training set, so we need the "labels" which are
        # called "output" here. The outputs are used to compute the loss. The
        # loss for us are the residuals. We have 4 residuals when it comes to
        # the spatial boundaries: x=0 for fluid phase, x=0 for solid phase,
        # x=1 for fluid phase, x=1 for solid phase

        # Let's compute the labels/outputs

        output_sb_0 = torch.zeros([input_sb.shape[0], 2])
        #
        # x = 0, fluid => f(t) = T_f(x=0, t) = ...
        #
        f = lambda t : ( ( self.T_hot - self.T_initial ) / ( 1.0 + np.exp(-200.0 * (t - 0.25))) ) + self.T_initial

        # Store fluid phase for x=0 outputs in the 2nd column
        output_sb_0[:,1] = f(input_sb[:,0])

        #
        # x = 0, solid => dT_s/dx|_(x=0) = 0
        #

        # output_sb_0[:,0] is already all 0.0, so nothing to do.


        output_sb_L = torch.zeros([input_sb.shape[0], 2])
        #
        # x = 1, fluid => dT_p/dx|_(x=1) = 0
        #

        # output_sb_L[:,0] is already all 0.0, so nothing to do.
        

        #
        # x = 1, solid => dT_s/dx|_(x=1) = 0
        #

        # output_sb_L[:,0] is already all 0.0, so nothing to do.

        #output_sb_0 = torch.zeros((input_sb.shape[0], 1))
        #output_sb_L = torch.zeros((input_sb.shape[0], 1))

        input_sb = torch.cat([input_sb_0, input_sb_L], 0)
        output_sb = torch.cat([output_sb_0, output_sb_L], 0)

        """
            input_sb:
                1st column: time
                2nd column: spatial

                1st 64 rows: x=0
                2bd 64 rows: x=1 
        
            output_sb:
                1st column: time
                2nd column: spatial

                1st 64 rows: x=0
                2bd 64 rows: x=1 
        """

        return input_sb, output_sb
    
    #  Function returning the input-output tensor required to assemble the
    #  training set S_int corresponding to the interior domain where the PDE is
    #  enforced
    def add_interior_points(self):
        # TODO: Implement
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 2))
        return input_int, output_int

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_sb, output_sb = self.add_spatial_boundary_points()  # S_sb
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_int, output_int = self.add_interior_points()  # S_int

        training_set_sb = DataLoader(
            torch.utils.data.TensorDataset(input_sb, output_sb),
            batch_size=2 * self.space_dimensions * self.n_sb,
            shuffle=False,
        )
        training_set_tb = DataLoader(
            torch.utils.data.TensorDataset(input_tb, output_tb),
            batch_size=self.n_tb,
            shuffle=False,
        )
        training_set_int = DataLoader(
            torch.utils.data.TensorDataset(input_int, output_int),
            batch_size=self.n_int,
            shuffle=False,
        )

        return training_set_sb, training_set_tb, training_set_int

    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual
    def apply_initial_condition(self, input_tb):
        # Note: We need to use our approximate_solution i.e. our NN because
        # it represents the actual function! And we need the function to
        # apply the initial conditions, since we have a PDE!
        # TODO: Implement

        u_pred_tb = self.approximate_solution(input_tb)
        return u_pred_tb

    # Function to compute the terms required in the definition of the SPATIAL boundary residual

    def applyboundary_condition(): 
        input_sb.requires_grad = True
        u_pred_sb = self.approximate_solution(input_sb)
        grad_u_x_solid = torch.autograd.grad(u_pred_sb[:,0].sum(), input_sb, create_graph = True)[0][:, 1] 
        grad_u_x_fluid = torch.autograd.grad(u_pred_sb[:,1].sum(), input_sb, create_graph=True)[0][:,1]

        spatial_fluid = u_pred_sb[:,1] 
        spatial_fluid[n_sb:] = grad_u_x_fluid[n_sb:]

        return grad_u_x_solid, spatial_fluid
        return u_pred_sb
    
   # Function to compute the terms required in the definition of the SPATIAL boundary residual
    def apply_boundary_conditions(self, input_sb):
        input_sb.requires_grad = True
        u_pred_sb = self.approximate_solution(input_sb)
        grad_u_x_solid = torch.autograd.grad(u_pred_sb[:,0].sum(), input_sb, create_graph = True)[0][:, 1] 
        grad_u_x_fluid = torch.autograd.grad(u_pred_sb[:,1].sum(), input_sb, create_graph = True)[0][:, 1]
        
        # Get f(t), which is stored in the outputs, for x=0 and the grad for x=1
        u_pred_sb_fluid = torch.cat([u_pred_sb[0:self.n_sb,1], grad_u_x_fluid[self.n_sb:]], dim=0)
        
        # Not much to do. We "apply" our BC to it.
        u_pred_sb_solid = grad_u_x_solid

        u_pred_sb = torch.cat([u_pred_sb_solid.reshape(-1,1), u_pred_sb_fluid.reshape(-1,1)], 1)

        return u_pred_sb
    
    # Function to compute the PDE residuals
    """
    def compute_pde_residual(self, input_int):
        # TODO: Implement
        input_int.requires_grad = True
        u = self.approximate_solution(input_int)
        u_sq = u * u

        # grad compute the gradient of a "SCALAR" function L with respect to some input nxm TENSOR Z=[[x1, y1],[x2,y2],[x3,y3],...,[xn,yn]], m=2
        # it returns grad_L = [[dL/dx1, dL/dy1],[dL/dx2, dL/dy2],[dL/dx3, dL/dy3],...,[dL/dxn, dL/dyn]]
        # Note: pytorch considers a tensor [u1, u2,u3, ... ,un] a vectorial function
        # whereas sum_u = u1 + u2 + u3 + u4 + ... + un as a "scalar" one

        # In our case ui = u(xi), therefore the line below returns:
        # grad_u = [[dsum_u/dx1, dsum_u/dy1],[dsum_u/dx2, dsum_u/dy2],[dsum_u/dx3, dL/dy3],...,[dsum_u/dxm, dsum_u/dyn]]
        # and dsum_u/dxi = d(u1 + u2 + u3 + u4 + ... + un)/dxi = d(u(x1) + u(x2) u3(x3) + u4(x4) + ... + u(xn))/dxi = dui/dxi
        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_int, create_graph=True)[
            0
        ][:, 1]

        grad_u_sq_x = torch.autograd.grad(u_sq.sum(), input_int, create_graph=True)[0][
            :, 1
        ]

        residual = grad_u_t - grad_u_xx
        return residual.reshape(
            -1,
        )
    """

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
        u = self.approximate_solution(input_int)

        # grad compute the gradient of a "SCALAR" function L with respect to some input nxm TENSOR Z=[[x1, y1],[x2,y2],[x3,y3],...,[xn,yn]], m=2
        # it returns grad_L = [[dL/dx1, dL/dy1],[dL/dx2, dL/dy2],[dL/dx3, dL/dy3],...,[dL/dxn, dL/dyn]]
        # Note: pytorch considers a tensor [u1, u2,u3, ... ,un] a vectorial function
        # whereas sum_u = u1 + u2 + u3 + u4 + ... + un as a "scalar" one

        # In our case ui = u(xi), therefore the line below returns:
        # grad_u = [[dsum_u/dx1, dsum_u/dy1],[dsum_u/dx2, dsum_u/dy2],[dsum_u/dx3, dL/dy3],...,[dsum_u/dxm, dsum_u/dyn]]
        # and dsum_u/dxi = d(u1 + u2 + u3 + u4 + ... + un)/dxi = d(u(x1) + u(x2) u3(x3) + u4(x4) + ... + u(xn))/dxi = dui/dxi
        
        grad_u_solid = torch.autograd.grad(u[:,0].sum(), input_int, create_graph=True)[0]
        grad_u_t_solid = grad_u_solid[:, 0] #solid phase first then fluid phase 
        grad_u_x_solid = grad_u_solid[:, 1]
        grad_u_fluid = torch.autograd.grad(u[:,1].sum(), input_int, create_graph=True)[0]
        grad_u_t_fluid = grad_u_fluid[:, 0] #solid phase first then fluid phase 
        grad_u_x_fluid = grad_u_fluid[:, 1]
        grad_u_xx_solid = torch.autograd.grad(grad_u_x_solid.sum(), input_int, create_graph=True)[0][:, 1]
        grad_u_xx_fluid = torch.autograd.grad(grad_u_x_fluid.sum(), input_int, create_graph=True)[0][:, 1]
        
        residual_solid = grad_u_t_solid - self.alpha_solid*grad_u_xx_solid - self.h_solid*(u[:,1] - u[:,0]) 
        residual_fluid = grad_u_t_fluid - self.alpha_fluid*grad_u_xx_fluid + self.h_fluid*(u[:,1] - u[:,0]) + self.U_fluid*grad_u_x_fluid
        
        return torch.cat([residual_solid.reshape(-1,1), residual_fluid.reshape(-1,1)], axis=1) #residual.reshape(-1, )


    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(
        self,
        inp_train_sb,
        u_train_sb,
        inp_train_tb,
        u_train_tb,
        inp_train_int,
        verbose=True
    ):
        global kk
        kk += 1
        # inp_train_tb.shape = [62, 2]
        u_pred_sb = self.apply_boundary_conditions(inp_train_sb)

        # inp_train_
        u_pred_tb = self.apply_initial_condition(inp_train_tb)

        assert u_pred_sb.shape[1] == u_train_sb.shape[1]
        assert u_pred_tb.shape[1] == u_train_tb.shape[1]


        # Compute residuals
        r_int = self.compute_pde_residual(inp_train_int)
        r_sb = u_train_sb - u_pred_sb
        r_tb = u_train_tb - u_pred_tb

        loss_sb = torch.mean(abs(r_sb) ** 2)
        loss_tb = torch.mean(abs(r_tb) ** 2)
        loss_int = torch.mean(abs(r_int) ** 2)

        loss_u = loss_sb + loss_tb

        loss = torch.log10(self.lambda_u * (loss_sb + loss_tb) + loss_int)
        if verbose:
            print(
                f"{kk} Total loss: ",
                round(loss.item(), 4),
                "| PDE Loss: ",
                round(torch.log10(loss_u).item(), 4),
                "| Function Loss: ",
                round(torch.log10(loss_int).item(), 4),
            )

        return loss

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()

        global kk
        kk = 0
        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose:
                print(
                    "################################ ",
                    epoch,
                    " ################################",
                )

            for j, (
                (inp_train_sb, u_train_sb),
                (inp_train_tb, u_train_tb),
                (inp_train_int, u_train_int),
            ) in enumerate(
                zip(self.training_set_sb, self.training_set_tb, self.training_set_int)
            ):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(
                        inp_train_sb,
                        u_train_sb,
                        inp_train_tb,
                        u_train_tb,
                        inp_train_int,
                        verbose=verbose
                    )
                    loss.backward()


                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print("Final Loss: ", history[-1])

        return history

    ################################################################################################
    
    def plotting(self):
        inputs = self.soboleng.draw(200000)
        inputs = self.convert(inputs)

        output = self.approximate_solution(inputs)
        output = output.reshape(-1,)

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)

        breakpoint()
        im2 = axs[1].scatter(
            inputs[:, 1].detach(), inputs[:, 0].detach(), c=output.detach(), cmap="jet"
        )
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title("Exact Solution")
        axs[1].set_title("Approximate Solution")

        plt.show()

