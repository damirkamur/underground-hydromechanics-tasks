from classes import *

task = Task(10000, 0.001, True)
# task.generate_regular_grid()
task.generate_log_grid()
task.make_matrix()
task.solve_matrix()
task.solve_error()
task.solve_u()
# task.show_u()
# task.show_p()
# task.save_p_plot()
# task.save_u_plot()
print(task.p_analit[0], task.p[0])