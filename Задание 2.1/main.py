from classes import *

task = Task(1000, 0.001, True)
task.generate_regular_grid()
# task.generate_log_grid()
task.make_matrix()
task.solve_matrix()
task.solve_error()
task.solve_u()
task.save_p_plot()
# task.save_u_plot()

R, rw, k0, m, dp, mu = 100, 0.1, 1e-12, 0.2, 1e6, 1e-3
T = sum([mu * (R - rw) * m / task.u[i] / k0 / dp * task.spaces[i] * (R - rw) for i in range(task.N)])

with open(f'results.txt', 'w', encoding='utf-8') as file:
    file.write(f'Время прохождения частицы между галереями: {T}')
