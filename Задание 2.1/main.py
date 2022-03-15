from classes import *
from matplotlib import pyplot as plt

x = list(range(10, 1051,50))
E1 = list()
E2 = list()
E3 = list()

for N in range(10, 1051,50):
    task = Task(N, 0.001, False)
    task.generate_regular_grid()
    task.make_matrix()
    task.solve_matrix()
    task.solve_u()
    task.solve_error()
    E1.append(task.E)
for N in range(10, 1051,50):
    task = Task(N, 0.001, True)
    task.generate_regular_grid()
    task.make_matrix()
    task.solve_matrix()
    task.solve_u()
    task.solve_error()
    E2.append(task.E)
for N in range(10, 1051,50):
    task = Task(N, 0.001, True)
    task.generate_log_grid()
    task.make_matrix()
    task.solve_matrix()
    task.solve_u()
    task.solve_error()
    E3.append(task.E)
plt.figure(1)
# plt.plot(x,E1)
plt.plot(x,E2)
plt.plot(x,E3)
plt.legend(('2','3'))
plt.minorticks_on()
plt.grid(which='major', linewidth=1)
plt.grid(which='minor', linestyle=':')
plt.grid(which='minor', linestyle=':')
plt.show()
# R, rw, k0, m, dp, mu = 100, 0.1, 1e-12, 0.2, 1e6, 1e-3
# T = mu * m * math.log(R / rw) * (R ** 2 - rw ** 2) / 2 / k0 / (task.p[task.N] - task.p[0]) / dp
#
# with open(f'results.txt', 'w', encoding='utf-8') as file:
#     file.write(f'Время прохождения частицы между галереями: {T}')
