import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import scipy.integrate as spi
# display
from IPython.display import display, Markdown, clear_output
# widget packages
import ipywidgets as widgets 
from ipywidgets import interact

from EPI_model import EPI_model


class Simple_SIR(EPI_model):
    def __init__(
        self, 
        inits = {'S':0.9, 'I':0.1, 'R':0}, 
        params = {'beta':0.1, 'gamma':0.1, 'N':10, 't_end':100, 'step_size':1}
    ):
        super().__init__("Simple SIR", inits, params)
    
    
    def diff_eqs(self, INPUT, t):
        Y=np.zeros((3))
        V = INPUT    
        Y[0] = - self.params['beta'] * V[0] * V[1]
        Y[1] = self.params['beta'] * V[0] * V[1] - self.params['gamma'] * V[1]
        Y[2] = self.params['gamma'] * V[1]
        return Y   # For odeint 
    
    def run(self):
        self.step = 0
        INPUT = tuple(self.inits.values())
        t_range = np.arange(0, self.params['t_end'] + self.params['step_size'], self.params['step_size'])
        self.all_sotw_vals = np.array(spi.odeint(self.diff_eqs,INPUT, t_range))
        self.all_sotw = np.round(self.params['N']*self.all_sotw_vals)
        self.dr = (-np.vstack(([self.params['N'],0,0], self.all_sotw[:-1,:])) + self.all_sotw)[:,1:]
        self.dr[self.dr < 0] = 0
        
    def current_sotw(self):
        return self.all_sotw[self.step], self.dr[self.step]
    
    def next_sotw(self):
        self.step += 1
        return self.current_sotw()
    
    def previous_sotw(self):
        self.step -= 1
        return self.current_sotw()

    def _interact(self, beta, gamma, S, I, R):
        params = self.params
        params['beta'], params['gamma'] = beta, gamma
        self.re_init(inits = {'S':S, 'I':I, 'R':R}, params = params)
        self.run()
        plt.figure(figsize = (15,10))
        plt.plot(self.all_sotw_vals[:,0], label = 'susceptibles')
        plt.plot(self.all_sotw_vals[:,1], label = 'infected')
        plt.plot(self.all_sotw_vals[:,2], label = 'removed')
        mx = round(np.max(self.all_sotw_vals[:,1]), 3)
        plt.axhline(y=mx, color='r', linestyle='-', label = f"maximum propotion of infected ppl : {mx}")
        plt.title(self.name)
        plt.legend()
        # plt.legend(iter(lines), ['susceptibles', 'infected', 'removed'])
    
    def interact(self):
        interact(self._interact, beta=(0, 0.5, 0.01), gamma = (0, 0.5, 0.01), S = (0, 1, 0.1), I = (0, 1, 0.1), R = (0, 1, 0.1))
        