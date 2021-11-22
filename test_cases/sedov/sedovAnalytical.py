#!/usr/bin/env python
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class SedovSolution(object):
    """
    see: [The Sedov self-similar point blast solutions in nonuniform media](https://link.springer.com/content/pdf/10.1007/BF01414626.pdf)

    rho0 = A*r**(-w)
    R_s = ((e * t**2)/(alpha * A))**(1/(nu + 2 - w))
    """
    def __init__(self, e, rho, gamma=4/3., nu=3, w=0., epsilon=1e-50):

        # w = 0 --> uniform background

        if not any(nu == np.array([1, 2, 3])):
            raise ValueError("nu (dimension of problem) need to be 1, 2 or 3!")
        
        self._epsilon = epsilon
        self._e = e
        self._gamma = gamma

        self._rho0 = rho
        self._rho1 = ((gamma + 1.)/(gamma - 1.)) * rho

        self._nDim = nu
        self._w = w

        # Constants for the parametric equations:
        self.w1 = (3*nu - 2 + gamma*(2-nu))/(gamma + 1.)
        self.w2 = (2.*(gamma-1) + nu)/gamma
        self.w3 = nu*(2.-gamma)

        self.b0 = 1./(nu*gamma - nu + 2)
        self.b2 = (gamma-1.)/(gamma*(self.w2-w))
        self.b3 = (nu-w)/(float(gamma)*(self.w2-w))
        self.b5 = (2.*nu-w*(gamma+1))/(self.w3-w)
        self.b6 = 2./(nu+2-w)
        self.b1 = self.b2 + (gamma+1.)*self.b0 - self.b6
        self.b4 = self.b1*(nu-w)*(nu+2.-w)/(self.w3-w)
        self.b7 = w*self.b6
        self.b8 = nu*self.b6

        self.c0 = 2*(nu-1)*np.pi + (nu-2)*(nu-3)  # simple interpolation of correct function (only for nu=1,2,3)
        self.c5 = 2./(gamma - 1)
        self.c6 = (gamma + 1)/2.
        self.c1 = self.c5*gamma
        self.c2 = self.c6/gamma
        self.c3 = (nu*gamma - nu + 2.)/((self.w1-w)*self.c6)
        self.c4 = (nu + 2. - w)*self.b0*self.c6

        # Characterize the solution
        f_min = self.c2 if self.w1 > w else self.c6

        f = np.logspace(np.log10(f_min), 0, 1e5)

        # Sort the etas for our interpolation function
        eta = self.parametrized_eta(f)
        f = f[eta.argsort()]
        eta.sort()

        d = self.parametrized_d(f)
        p = self.parametrized_p(f)
        v = self.parametrized_v(f)

        # If min(eta) != 0 then all values for eta < min(eta) = 0
        if eta[0] > 0:
            e01 = [0., eta[0]*(1-1e-10)]
            d01 = [0., 0]
            p01 = [0., 0]
            v01 = [0., 0]

            eta = np.concatenate([np.array(e01), eta])
            d = np.concatenate([np.array(d01), d])
            p = np.concatenate([np.array(p01), p])
            v = np.concatenate([np.array(v01), v])

        # Set up our interpolation functions
        self._d = interp1d(eta, d, bounds_error=False, fill_value=1./self._rho1)
        self._p = interp1d(eta, p, bounds_error=False, fill_value=0.)
        self._v = interp1d(eta, v, bounds_error=False, fill_value=0.)

        # Finally Calculate the normalization of R_s:
        integral = eta**(nu-1)*(d*v**2 + p)
        integral = 0.5 * (integral[1:] + integral[:-1])
        d_eta = (eta[1:] - eta[:-1])

        # calculate integral and multiply by factor
        alpha = (integral*d_eta).sum() * (8*self.c0)/((gamma**2-1.)*(nu+2.-w)**2)
        self._c = (1./alpha)**(1./(nu+2-w))

    def parametrized_eta(self, var):
        return (var**-self.b6)*((self.c1*(var-self.c2))**self.b2)*((self.c3*(self.c4-var))**(-self.b1))

    def parametrized_d(self, var):
        return (var**-self.b7)*((self.c1*(var-self.c2))**(self.b3-self._w*self.b2)) * \
                    ((self.c3*(self.c4-var))**(self.b4+self._w*self.b1))*((self.c5*(self.c6-var))**-self.b5)

    def parametrized_p(self, var):
        return (var**self.b8)*((self.c3*(self.c4-var))**(self.b4+(self._w-2)*self.b1)) * \
                    ((self.c5*(self.c6-var))**(1-self.b5))

    def parametrized_v(self, var):
        return self.parametrized_eta(var) * var

    # Shock properties
    def shock_radius(self, t):
        # outer radius at time t
        t = np.maximum(t, self._epsilon)
        return self._c * (self.e*t**2/self.rho0)**(1./(self._nDim + 2-self._w))

    def shock_velocity(self, t):
        # velocity of the shock wave
        t = np.maximum(t, self._epsilon)
        return (2./(self._nDim+2-self._w)) * self.shock_radius(t) / t

    def post_shock_pressure(self, t):
        # post shock pressure
        return (2./(self.gamma+1))*self.rho0*self.shock_velocity(t)**2

    @property
    def post_shock_density(self, t=0):
        # post shock density
        return self._rho1

    def rho(self, r, t):
        # density at radius r and time t
        eta = r/self.shock_radius(t) 
        return self.post_shock_density*self._d(eta)

    def pressure(self, r, t):
        # pressure at radius r and time t
        eta = r/self.shock_radius(t) 
        return self.post_shock_pressure(t)*self._p(eta)

    def velocity(self, r, t):
        # velocity at radius r, and time t
        eta = r/self.shock_radius(t) 
        return self._v(eta)*(2/(self.gamma+1))*self.shock_velocity(t)

    def internal_energy(self, r, t):
        # internal energy at radius r and time t
        return self.pressure(r, t)/(self.rho(r, t)*(self.gamma-1))

    def entropy(self, r, t):
        # entropy at radius, r, and time, t
        return self.pressure(r, t)/self.rho(r, t)**self.gamma

    # Other properties
    @property
    def e(self):
        # total energy
        return self._e

    @property
    def gamma(self):
        # ratio of specific heats
        return self._gamma

    @property
    def rho0(self):
        # background density
        return self._rho0


class Sedov(object):
    """
    Analytical solution for the sedov blast wave problem
    """
    def __init__(self, time, r_max):

        rho0 = 1.0  #1
        e0 = 1.0 #1e5
        gamma = 5/3. #1.66 #1.333
        w = 0  # Power law index
        n_dim = 3

        self.sol = SedovSolution(e0, rho0, gamma=gamma, w=w, nu=n_dim)
        self.r = np.linspace(0, r_max, 1001)[1:]
        self.t = time

        print("Shock radius: {}".format(self.sol.shock_radius(self.t)))

    def compute(self, y):
        return map(self.determine, ['r', y])

    def determine(self, x):
        if x == 'r':
            return self.r
        elif x == 'velocity':
            return self.sol.velocity(self.r, self.t)
        elif x == 'rho':
            return self.sol.rho(self.r, self.t)
        elif x == 'pressure':
            return self.sol.pressure(self.r, self.t)
        elif x == 'internal_energy':
            return self.sol.internal_energy(self.r, self.t)
        else:
            raise AttributeError("Sedov solution for variable %s not known"%x)


if __name__ == '__main__':

    #sedov = Sedov(0.001, 1)
    sedov = Sedov(0.02, 1)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    ax1.plot(*sedov.compute('rho'), label="rho")
    ax2.plot(*sedov.compute('pressure'), label="pressure")
    ax3.plot(*sedov.compute('internal_energy'), label="internal energy")
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    plt.show()
