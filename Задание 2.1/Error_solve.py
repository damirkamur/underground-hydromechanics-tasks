from classes import *

N = [10 + i for i in range(0, 991, 10)]
errors = np.zeros(len(N))
for i, n in enumerate(N):
    task = Task(n, 0.001)
    task.generate_regular_grid()
    task.make_matrix()
    task.solve_matrix()
    task.solve_error()
    errors[i] = task.E
plt.plot(N, errors, 'r')
plt.title('Погрешность давления')
plt.minorticks_on()
plt.grid()
plt.xlabel('N')
plt.ylabel('E')
plt.minorticks_on()
plt.grid(which='major', linewidth=1)
plt.grid(which='minor', linestyle=':')
plt.grid(which='minor', linestyle=':')
plt.savefig('Погрешность от сетки (регулярная сетка)', dpi=300)
