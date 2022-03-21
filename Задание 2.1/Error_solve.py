import matplotlib.pyplot as plt

from classes import *

N = list(range(10, 1000, 20))
errors1 = np.zeros(len(N))
errors2 = np.zeros(len(N))
errors3 = np.zeros(len(N))
for i, n in enumerate(N):
    task = Task(n, 0.001)
    task.generate_regular_grid()
    task.make_matrix()
    task.solve_matrix()
    task.solve_error()
    errors1[i] = task.E
for i, n in enumerate(N):
    task = Task(n, 0.001, True)
    task.generate_regular_grid()
    task.make_matrix()
    task.solve_matrix()
    task.solve_error()
    errors2[i] = task.E
for i, n in enumerate(N):
    task = Task(n, 0.001)
    task.generate_log_grid()
    task.make_matrix()
    task.solve_matrix()
    task.solve_error()
    errors3[i] = task.E
plt.plot(N, errors1, 'r')
plt.plot(N, errors2, 'g')
plt.plot(N, errors3, 'b')
plt.title('Погрешность давления')
plt.legend(('Р.С. без поправки)', 'Р.С. с поправкой', 'Л.С.'))
plt.minorticks_on()
plt.grid()
plt.xlabel('N')
plt.ylabel('E')
plt.minorticks_on()
plt.grid(which='major', linewidth=1)
plt.grid(which='minor', linestyle=':')
plt.grid(which='minor', linestyle=':')
plt.savefig('Погрешность от сетки', dpi=300)
