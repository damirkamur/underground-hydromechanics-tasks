from classes import *

task = Task(1000, 0.001)
task.generate_log_grid()
task.make_matrix()
task.solve_matrix()
task.solve_error()
task.solve_u()

R, rw, k0, m, dp, mu = 100, 0.1, 1e-12, 0.2, 1e6, 1e-3
T = mu * m * math.log(R / rw) * (R ** 2 - rw ** 2) / 2 / k0 / (task.p[task.N] - task.p[0]) / dp

with open(f'results.txt', 'w', encoding='utf-8') as file:
    file.write(f'Время прохождения частицы между галереями: {T}')
