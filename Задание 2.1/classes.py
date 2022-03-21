import math

import numpy as np
from matplotlib import pyplot as plt
from math import sqrt


class Task:
    def __init__(self, N: int, rw: float, correction_coef: bool = False):
        if N <= 0 or rw >= 1.0 or rw <= 0:
            raise Exception('Некорректные входные данные')
        self.__N = N
        self.__rw = rw
        self.__A = None
        self.__b = None
        self.__x = None
        self.__spaces = None
        self.__p = None
        self.__p_analit = None
        self.__E = None
        self.__u = None
        self.__u_analit = None
        self.__correction_coef = correction_coef

    def generate_regular_grid(self):
        self.__x = np.linspace(self.rw, 1.0, self.N + 1)
        self.__generate_grid_spaces()
        self.__solve_p_analit()
        self.__solve_u_analit()

    def generate_log_grid(self):
        self.__x = np.zeros(self.N + 1)
        coef = 1.0 / self.rw
        for i in range(self.N + 1):
            self.__x[i] = self.rw * coef ** (i / self.N)
        self.__generate_grid_spaces()
        self.__solve_p_analit()
        self.__solve_u_analit()

    def __generate_grid_spaces(self):
        self.__spaces = np.zeros(self.N)
        for i in range(self.N):
            self.__spaces[i] = self.x[i + 1] - self.x[i]
        if self.__correction_coef:
            self.__tetta = np.ones(self.N + 1)
            for i in range(self.N):
                self.__tetta[i] = 2 * (self.x[i + 1] - self.x[i]) / (self.x[i + 1] + self.x[i]) / math.log(
                    self.x[i + 1] / self.x[i])
        else:
            self.__tetta = np.ones(self.N + 1)

    def make_matrix(self):
        self.__A = np.zeros((self.N + 1, self.N + 1))
        self.__b = np.zeros(self.N + 1)
        for i in range(1, self.N):
            eq_ind = i - 1
            self.__A[eq_ind][i - 1] = -(self.x[i] + self.x[i - 1]) * self.tetta[i - 1] / 2 / self.spaces[i - 1]
            self.__A[eq_ind][i] = (self.x[i + 1] + self.x[i]) * self.tetta[i] / 2 / self.spaces[i] + (
                    self.x[i] + self.x[i - 1]) * self.tetta[i - 1] / 2 / self.spaces[i - 1]
            self.__A[eq_ind][i + 1] = -(self.x[i + 1] + self.x[i]) * self.tetta[i] / 2 / self.spaces[i]
        eq_ind = self.N - 1
        self.__A[eq_ind][0] = 1
        eq_ind += 1
        self.__A[eq_ind][self.N] = 1
        self.__b[eq_ind] = 1

    def solve_matrix(self):
        self.__p = np.linalg.solve(self.A, self.b)

    def show_p(self):
        plt.figure(1)
        plt.plot(self.x, self.p_analit, 'b', self.x, self.p, 'r--')
        plt.title('Функция давления')
        plt.minorticks_on()
        plt.grid()
        plt.legend(('Точное решение', 'Численное решение'))
        plt.xlabel('x')
        plt.ylabel('p')
        plt.minorticks_on()
        plt.grid(which='major', linewidth=1)
        plt.grid(which='minor', linestyle=':')
        plt.grid(which='minor', linestyle=':')
        plt.show()

    def save_p_plot(self):
        plt.figure(2)
        plt.plot(self.x, self.p_analit, 'b', self.x, self.p, 'r--')
        plt.title('Функция давления')
        plt.minorticks_on()
        plt.grid()
        plt.legend(('Точное решение', 'Численное решение'))
        plt.xlabel('x')
        plt.ylabel('p')
        plt.minorticks_on()
        plt.grid(which='major', linewidth=1)
        plt.grid(which='minor', linestyle=':')
        plt.grid(which='minor', linestyle=':')
        plt.savefig(f'Функция давления (N={self.N})', dpi=300)

    def __solve_p_analit(self):
        self.__p_analit = np.zeros(self.N + 1)
        coef = -1 / math.log(self.rw)
        for i in range(self.N + 1):
            self.__p_analit[i] = math.log(self.x[i])
        self.__p_analit *= coef
        self.__p_analit += 1

    def __solve_u_analit(self):
        self.__u_analit = np.zeros(self.N + 1)
        coef = 1 / math.log(self.rw)
        for i in range(self.N + 1):
            self.__u_analit[i] = 1 / self.x[i]
        self.__u_analit *= coef

    def solve_error(self):
        self.__E = math.sqrt(sum([(self.p[i] - self.p_analit[i]) ** 2 for i in range(self.N + 1)]) / (self.N + 1))

    def solve_u(self):
        self.__u = np.zeros(self.N + 1)
        for i in range(self.N):
            self.__u[i] = (self.p[i] - self.p[i + 1]) / self.spaces[i]
        self.__u[self.N] = (self.p[self.N - 1] - self.p[self.N]) / self.spaces[self.N - 1]

    def show_u(self):
        plt.figure(3)
        plt.plot(self.x, self.u_analit, 'b', self.x, self.u, 'r--')
        plt.title('Функция скорости фильтрации')
        plt.minorticks_on()
        plt.grid()
        plt.legend(('Точное решение', 'Численное решение'))
        plt.xlabel('x')
        plt.ylabel('u')
        plt.minorticks_on()
        plt.grid(which='major', linewidth=1)
        plt.grid(which='minor', linestyle=':')
        plt.grid(which='minor', linestyle=':')
        plt.show()

    def save_u_plot(self):
        plt.figure(4)
        plt.plot(self.x, self.u_analit, 'b', self.x, self.u, 'r--')
        plt.title('Функция скорости фильтрации')
        plt.minorticks_on()
        plt.grid()
        plt.legend(('Точное решение', 'Численное решение'))
        plt.xlabel('x')
        plt.ylabel('u')
        plt.minorticks_on()
        plt.grid(which='major', linewidth=1)
        plt.grid(which='minor', linestyle=':')
        plt.grid(which='minor', linestyle=':')
        plt.savefig(f'Функция скорости фильтрации (N={self.N})', dpi=300)

    @property
    def N(self):
        return self.__N

    @property
    def rw(self):
        return self.__rw

    @property
    def x(self):
        return self.__x

    @property
    def spaces(self):
        return self.__spaces

    @property
    def A(self):
        return self.__A

    @property
    def b(self):
        return self.__b

    @property
    def p(self):
        return self.__p

    @property
    def p_analit(self):
        return self.__p_analit

    @property
    def E(self):
        return self.__E

    @property
    def u(self):
        return self.__u

    @property
    def u_analit(self):
        return self.__u_analit

    @property
    def tetta(self):
        return self.__tetta
