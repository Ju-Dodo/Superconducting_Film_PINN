from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch as tr
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as clrs
import scipy.integrate as integ
import mpmath as mp
import pandas as pd

tr.manual_seed(13985)

tr.cuda.device(0)
print(tr.cuda.current_device())


class PINN(nn.Module):

    def __init__(self, learning_rate, beta_params, PDE_params):
        super(PINN, self).__init__()

        #Define the physical parameters of the problem
        self.E_c = PDE_params[0] #Electric Field Criterion
        self.B_0 = PDE_params[1] #Applied AC Field
        self.n = PDE_params[2] #nonlinearity of resistance
        self.asp = PDE_params[3] #aspect ratio
        self.alph = PDE_params[4] #anisotropic resistivity

        #Define the layers of the Neural Network
        self.lin1 = nn.Linear(3, 10)
        self.lin2 = nn.Linear(10, 10)
        self.lin3 = nn.Linear(10, 5)
        self.lin4 = nn.Linear(5, 5)
        self.lin5 = nn.Linear(5, 1)

        # Define the optimizer, loss function and scaler
        self.optimizer  = tr.optim.Adam(self.parameters(),lr=1E-4)
        self.LossCrit = tr.nn.MSELoss()
        self.scaler = tr.cuda.amp.GradScaler()

    def M(self, x):
        # The Magnetisation as a function of the input, x
        x = tr.sin(self.lin1(x))
        x = tr.tanh(self.lin2(x))
        x = tr.tanh(self.lin3(x))
        x = tr.tanh(self.lin4(x))
        x = self.lin5(x)

        return x

    def d2M(self, x):
        # The analytic second derivatives of the Magnetisation
        x1 = tr.sin(self.lin1(x))
        x2 = tr.tanh(self.lin2(x1))
        x3 = tr.tanh(self.lin3(x2))
        x4 = tr.tanh(self.lin4(x3))

        S1 = tr.cos(self.lin1(x))
        S2 = 1 - x2 * x2
        S3 = 1 - x3 * x3
        S4 = 1 - x4 * x4

        # Using Einstein Summation notation to multiply components together
        dS1 = tr.einsum("mn,vm->vmn", self.lin1.weight, -x1)

        VMN = tr.einsum("vm,mn->vmn", S1, self.lin1.weight)
        VLN = tr.einsum("lm,vmn->vln", self.lin2.weight, VMN)

        dS2 = tr.einsum("vl,vln->vln", -2 * S2 * x2, VLN)

        VLN_1 = tr.einsum("vl,vln->vln", S2, VLN)
        VKN = tr.einsum("kl,vln->vkn", self.lin3.weight, VLN_1)

        dS3 = tr.einsum("vk,vkn->vkn", -2 * S3 * x3, VKN)

        VKN_1 = tr.einsum("vk,vkn->vkn", S3, VKN)
        VJN = tr.einsum("jk,vkn->vjn", self.lin4.weight, VKN_1)

        dS4 = tr.einsum("vj,vjn->vjn", -2 * S4 * x4, VJN)

        pS2 = tr.einsum("ij,vj->vij", self.lin5.weight, S4)
        pS3 = tr.einsum("vij,jk->vik", pS2, self.lin4.weight)
        pS3 = tr.einsum("vik,vk->vik", pS3, S3)
        pS4 = tr.einsum("vik,kl->vil", pS3, self.lin3.weight)
        pS4 = tr.einsum("vil,vl->vil", pS4, S2)

        VJPN = tr.einsum("vjp,vjn->vjpn",dS4,VJN)
        T1 = tr.einsum("ij,vjpn->vipn", self.lin5.weight, VJPN)

        VKPN = tr.einsum("vkp,vkn->vkpn",dS3,VKN)
        T2 = tr.einsum("jk,vkpn->vjpn", self.lin4.weight, VKPN)
        T2 = tr.einsum("vij,vjpn->vipn", pS2, T2)

        VLPN = tr.einsum("vlp,vln->vlpn",dS2,VLN)
        T3 = tr.einsum("kl,vlpn->vkpn", self.lin3.weight, VLPN)
        T3 = tr.einsum("vik,vkpn->vipn", pS3, T3)

        T4 = tr.einsum("lm,vmp,mn->vlpn", self.lin2.weight, dS1, self.lin1.weight)
        T4 = tr.einsum("vil,vlpn->vipn", pS4, T4)

        dM_dx = T1 + T2 + T3 + T4
        dM_dx = tr.squeeze(dM_dx[:, 0:1, :, :], dim=1)

        return dM_dx

    def dM(self, x):
        # The analytic first derivatives of the magnetisation
        x1 = tr.sin(self.lin1(x))
        x2 = tr.tanh(self.lin2(x1))
        x3 = tr.tanh(self.lin3(x2))
        x4 = tr.tanh(self.lin4(x3))

        S1 = tr.cos(self.lin1(x))
        S2 = 1 - x2 * x2
        S3 = 1 - x3 * x3
        S4 = 1 - x4 * x4

        # Using Einstein Summation Notation to multiply components together
        VMN = tr.einsum("vm,mn->vmn", S1, self.lin1.weight)
        VLN = tr.einsum("lm,vmn->vln", self.lin2.weight, VMN)
        VLN = tr.einsum("vl,vln->vln", S2, VLN)
        VKN = tr.einsum("kl,vln->vkn", self.lin3.weight, VLN)
        VKN = tr.einsum("vk,vkn->vkn", S3, VKN)
        VJN = tr.einsum("jk,vkn->vjn", self.lin4.weight, VKN)
        VJN = tr.einsum("vj,vjn->vjn", S4, VJN)
        xcat = tr.einsum("ij,vjn->vin", self.lin5.weight, VJN)

        dA_dx = tr.squeeze(xcat[:, 0:1, :], dim=1)

        return dA_dx

    def PDE(self, X_f):
        # Define the partial differential equation that needs to be solved
        M_grad = self.dM(X_f)
        M_dblgrad = self.d2M(X_f)

        # The input vector, X_f, is the 2-d cartesian coordinates of the superconducting rectangle and the time coordinate (x,y,t)
        x = X_f[:, 0:1]
        y = X_f[:,1:2]
        t = X_f[:, 2:3]

        M_t = M_grad[:, 2:3]
        M_x = M_grad[:, 0:1]
        M_y = M_grad[:, 1:2]
        J_x = M_y
        J_y = -M_x

        J_sq = J_y**2 + (J_x/(self.alph*self.asp))**2

        M_xx = M_dblgrad[:, 0:1, 0:1]
        M_yy = M_dblgrad[:, 1:2, 1:2]
        M_xy = M_dblgrad[:, 0:1, 1:2]

        # M_PDE should always be zero if the Partial Differential Equation is satisfied (D(M) -f(M) = 0)
        M_PDE = M_t - self.B_0 * (tr.sin(t) + self.E_c * (tr.pow(J_sq,(self.n-1)/2) * (M_xx + M_yy/self.alph) + ((self.n-1)/2) * tr.pow(J_sq,(self.n-3)/2) * ((2 * M_x * M_xx + 2 * M_y * M_xy / (self.alph*self.asp)**2) * M_x + (2 * M_x * M_xy + 2 * M_y * M_yy / (self.alph * self.asp)**2) * M_y/self.alph)))

        return M_PDE

    def M_grad(self, X):

        """The first derivative of the magnetisation using autograd
        The output is the same as self.dM() but this function is slower
        This function is only here to check for consistency and accuracy"""
        M_J = []
        for x_t in X:
            M_J.append(tr.autograd.functional.jacobian(self.M, x_t, create_graph=True).squeeze())

        M_J = tr.stack(M_J)

        return M_J

    def M_dblgrad(self, X):
        """The second derivative of the magnetisation using autograd
        The output is the same as self.d2M() but this function is slower
        This function is only here to check for consistency and accuracy"""
        M_J = []
        for x_t in X:
            M_J.append(tr.autograd.functional.hessian(self.M, x_t, create_graph=True))
        M_J = tr.stack(M_J)

        return M_J

    def TrainCycle(self, lb, ub, nTrain, batch_size):
        """
        defines the training cycle for the neural network
        :param lb: defines the lower bounds (x_min, y_min, t_min) of the problem
        :param ub: defines the upper bounds (x_max, y_max, t_max) of the problem
        :param nTrain: the number of training iterations
        :param batch_size: the size of the training batch
        :return:
        """
        # define the optimiser and the loss lists
        self.optimizer = tr.optim.AdamW(self.parameters(),lr=1E-3)
        itNum = []
        itLossPDE = []
        itLossbc = []

        # the weights for the boundary conditions and partial differential equations
        w_pde = 1.0
        w_bc = 10.0

        for it in range(nTrain):
            if it==1000:
                # change the optimizer after 1000 iterations
                self.optimizer = tr.optim.AdamW(self.parameters(),lr=1E-4)
            if it % 10 == 0:

                # create a random sample from the problem space every 10 iterations
                X_f = tr.rand((batch_size, 3), dtype=tr.double).cuda() * (ub - lb) + lb

                # scale the inputs to ensure that more samples are taken in regions that are most rapidly changing
                X_f[:, 0:2] = 0.5 * (3 * X_f[:, 0:2] - X_f[:, 0:2]**3)
                x = X_f[:, 0:1]
                y = X_f[:, 1:2]

                X_ux = X_f.clone().detach()
                X_ux[:, 0:1] = X_ux[:, 0:1] * 0 + 1

                X_lx = X_f.clone().detach()
                X_lx[:, 0:1] = X_lx[:, 0:1] * 0

                X_uy = X_f.clone().detach()
                X_uy[:, 1:2] = X_uy[:, 1:2] * 0 + 1

                X_ly = X_f.clone().detach()
                X_ly[:, 1:2] = X_ly[:, 1:2] * 0

                X_ut = X_f.clone().detach()
                X_ut[:, 2:3] = X_ut[:, 2:3] * 0 + np.pi

                X_lt = X_f.clone().detach()
                X_lt[:, 2:3] = X_lt[:, 2:3] * 0

            self.optimizer.zero_grad()

            # find the difference between the Neural Network state and the required solution
            # M_PDE is zero when the PDE is satisfied
            M_PDE = self.PDE(X_f)

            # M_lbc and M_ubc are zero when the problem's boundary conditions are satisfied
            M_lbc = tr.cat((self.M(X_ux), self.dM(X_lx)[:,0:1], self.M(X_uy), self.dM(X_ly)[:,1:2], self.M(X_lt), self.dM(X_lt)[:,2:3]), 1)
            M_ubc = tr.cat((0 * self.M(X_ux), 0 * self.M(X_lx), 0 * self.M(X_uy), 0 * self.M(X_ly), -self.M(X_ut), -self.dM(X_ut)[:,2:3]), 1)

            # the Loss functions are defined for both PDE and boundary conditions
            lossPDE = self.LossCrit(M_PDE / (self.B_0), M_PDE * 0)
            lossbc = self.LossCrit(M_lbc / (self.B_0), M_ubc / (self.B_0))

            if it % 10 == 0:
                # record the loss every 10 iterations
                if it > 0:
                    itNum.append(np.log10(float(it)))
                    itLossPDE.append(np.log10(lossPDE.item()))
                    itLossbc.append(np.log10(lossbc.item()))
                print('at it. %d:  LossPDE=%.8e, lossbc=%.8e,lr at it. %d: %.2e' % (
                    it, lossPDE.item(), lossbc.item(), it, self.optimizer.param_groups[0]['lr']))

            # calculate the loss every iteration based on weightings and scalings
            loss = w_pde * lossPDE + w_pde * lossPDE**0.5 + w_bc * lossbc + w_bc * lossbc**0.5

            # update the model parameters
            loss.backward()
            self.optimizer.step()

        # at the end of the training cycle, save the loss to a file
        LFile = pd.DataFrame({'it#': itNum, 'Loss_PDE': itLossPDE, 'Loss_bc': itLossbc})
        LFile.to_csv("C:\\\\Users\\Andy\\OneDrive - Durham University\\PyTorch_NNs\\Loss_File_Train.csv")

    def InitCycle(self, lb, ub, nTrain, batch_size):
        """
        defines the initialisation cycle for the neural network
        :param lb: defines the lower bounds (x_min, y_min, t_min) of the problem
        :param ub: defines the upper bounds (x_max, y_max, t_max) of the problem
        :param nTrain: the number of training iterations
        :param batch_size: the size of the training batch
        :return:
        """

        """
        This initialisation routine is to train the neural network against a well known, analytic solution of the PDE
        By doing this, we can enforce the boundary contitions from an early stage. This aids with convergence
        when training the generalised model
        """
        # define the loss and iteration number list
        itNum = []
        itLoss0 = []
        for it in range(nTrain):
            if it % 10 == 0:
                # update the batch every 10 iterations
                X_f = tr.rand((batch_size, 3), dtype=tr.double).cuda() * (ub - lb) + lb
                x = X_f[:, 0:1]
                z = X_f[:, 1:2]
                t = X_f[:, 2:3]

                # train the model against an analytic target
                M_Target = ((3*x**2-3) * (3*z**2-3) * tr.cos(t) / 9.0).clone().detach()
            self.optimizer.zero_grad()

            M_pred = self.M(X_f)

            w_bc = 0.01

            # calculate the loss against the analytic target
            loss0 = self.LossCrit(M_pred, M_Target)

            # update the optimiser when the loss plateaus
            if it==300:
                self.optimizer = tr.optim.AdamW(self.parameters(),amsgrad=True,lr=1E-3)

            if it==630:
                self.optimizer.param_groups[0]['lr'] = 5E-4

            # calculate the gradients
            self.scaler.scale(loss0).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if it % 10 == 0:
                if it > 0:
                    itNum.append(np.log10(float(it)))
                    itLoss0.append(np.log10(loss0.item()))
                print('loss0 at it. %d:  %.8e, lr at it. %d: %.2e' % (
                    it, loss0.item(), it, self.optimizer.param_groups[0]['lr']))

        self.lin5.weight.data = self.lin5.weight.data * self.B_0
        self.lin5.bias.data = self.lin5.bias.data * self.B_0

        # save the loss file to a csv
        LFile = pd.DataFrame({'it#': itNum, 'Loss': itLoss0})
        LFile.to_csv("C:\\\\Users\\Andy\\OneDrive - Durham University\\PyTorch_NNs\\Loss_File_Init.csv")

    def J(self, X):
        # returns the current vector obtained from the neural network
        dM = self.dM(X)
        J_x = dM[:, 1:2]
        J_y = -dM[:, 0:1]

        return J_x, J_y

    def MagTot(self, t):
        # returns the total magnetisation
        t_prim=t%(2*np.pi)
        if t_prim<=np.pi:
            return 4 * integ.dblquad(lambda x, y: self.M(tr.tensor([x, y, t_prim],dtype=tr.double).cuda())[0], 0.0, 1.0, lambda x: 0.0, lambda x: 1.0)[0]
        else:
            return -4 * integ.dblquad(lambda x, y: self.M(tr.tensor([x, y, t_prim - np.pi],dtype=tr.double).cuda())[0], 0.0, 1.0, lambda x: 0.0, lambda x: 1.0)[0]


def PlotFrame(i, M, xarr, zarr):
    """
    Plots an idividual frame of a movie of the magnetisation
    """
    ax.cla()
    i = int(i)

    ax.pcolormesh(xarr, zarr, M[i], vmin=-1.0, vmax=1.0)
    ax.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylabel('Current / $GAm^{-2}$')


def Init_Frame():
    """
    initialised the movie frame
    """
    colormesh0 = ax.pcolormesh(xarr, zarr, Marr[0], vmin=-1.0, vmax=1.0)
    ax.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.colorbar(colormesh0, ax=ax)
    ax.set_ylabel('Current / $GAm^{-2}$')


MODEL_PATH = 'C:\\\\Users\\Andy\\OneDrive - Durham University\\PyTorch_NNs\\Base_Model.pth'

if __name__ == "__main__":
    # define the physical constraints of the problem space
    E_c = 1.024E-3
    asp = 1.0
    alph = 1.0
    lb = tr.tensor([0.0, 0.0, 0.0], dtype=tr.double).cuda()
    ub = tr.tensor([1.0, 1.0, np.pi], dtype=tr.double).cuda()

    tarr = np.linspace(0,2 * np.pi,201)
    Marr = np.zeros(len(tarr))

    # define the neural network as a material with linear resistivity
    model = PINN(1E-4, (0.99, 0.999), [E_c, 1.0, 1.0, asp, alph]).cuda()
    model.double()
    print(model)

    model.InitCycle(lb, ub, 10000, int(1024))
    # save the converged model
    tr.save(model,MODEL_PATH)


    """
    Train the model on a range of AC field strengths and nonlinearities
    """
    for B_0 in [1.0]:
        Barr = B_0 * np.cos(tarr)
        for n in [25.0]:

            modelT = tr.load(MODEL_PATH)
            modelT.lin5.weight.data = B_0 * modelT.lin5.weight.data
            modelT.lin5.bias.data = B_0 * modelT.lin5.bias.data
            modelT.B_0 = B_0
            modelT.n = n

            modelT.train()
            t_s = time.time()
            modelT.TrainCycle(lb, ub, 1000, int(256))
            modelT.eval()

            for i in range(len(tarr)):
                Marr[i] = modelT.MagTot(tarr[i])

            df = pd.DataFrame(data={'Time': tarr, 'Field': Barr, 'Mag': Marr})
            df.to_csv("C:\\\\Users\\Andy\\OneDrive - Durham University\\PyTorch_NNs\\Bsp=%0.7f,n=%0.3f, t=%0.4f.csv" % (B_0, n, time.time() - t_s))
            print("n=%0.3f,Bsp=%0.7f done" % (n, B_0))

    """
    Plots a video of the solution at 128x128 resolution over 64 frames
    """

    nx = 128
    nz = 128
    nt = 64
    N_f = nx * nz * nt

    xarr = np.linspace(-1.0, 1.0, nx)
    zarr = np.linspace(-1.0, 1.0, nz)
    tarr = np.linspace(0, 2 * np.pi, nt)

    X_T = np.meshgrid(xarr, zarr, tarr)
    X_T = tr.tensor(X_T).cuda()
    X_T = tr.flatten(X_T, start_dim=1).t()
    X_T = X_T.double()

    tr.cuda.empty_cache()

    Ms = tr.tensor([])
    i_start = 0
    while i_start < N_f:
        tr.cuda.empty_cache()
        x = X_T[i_start:i_start + nx * nz, :]
        with tr.no_grad():
            Ms = tr.cat((Ms, model.M(x).cpu() / B_0), 0)
        i_start = i_start + nx * nz

    Marr = Ms.reshape((nx, nz, nt))
    Marr = Marr.permute(2, 0, 1)

    Marr = Marr.detach().numpy()
    print(Marr)

    fig, ax = plt.subplots(figsize=(12, 7))
    frames = np.linspace(0, int(nt - 1), nt)
    ani = FuncAnimation(fig, PlotFrame, fargs=(Marr, xarr, zarr), frames=frames, init_func=Init_Frame, interval=22)
    ani.save("C:\\\\Users\\Andy\\OneDrive - Durham University\\TensorflowVids\\Test.mp4", fps=16,
             progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'))
