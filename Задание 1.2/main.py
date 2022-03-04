import numpy as np
from matplotlib import pyplot as plt

task_number = 1


def p_analit_1(x):
    return 1 - x


def u_analit_1(x):
    return 1


def p_analit_2(x):
    return 1 - 2 / 11 * x if x < 0.5 else 20 / 11 - 20 / 11 * x


def u_analit_2(x):
    return 2 / 11


def p_analit_3(x):
    return -2 / 11 * x + 1 if x < 0.75 else -38 / 11 * x + 38 / 11


def u_analit_3(x):
    return 2 / 11


def define_zone_num(xx):
    zone_num = 0
    for j in range(NN, 0, -1):
        if xx >= zone_coordinates[j]:
            zone_num = j
            break
    return zone_num if zone_num < NN else NN - 1


def define_type_x(xx):
    if abs(xx - x[0]) < 10e-15:
        return 'left'
    elif abs(xx - x[x.size - 1]) < 10e-15:
        return 'right'
    else:
        for i in range(zone_coordinates.size - 1):
            if abs(xx - zone_coordinates[i]) < 10e-15:
                return 'knot'
        else:
            return 'inside'


# Разбиение минимальной зоны
N_min = 2

# Кол-во зон и их границы
with open('zone_info.txt', 'r') as file:
    zone_coordinates = np.array(list(map(float, file.readline().split())))
    zone_f = np.array(list(map(float, file.readline().split())))
NN = len(zone_coordinates) - 1
zone_lengths = [zone_coordinates[i + 1] - zone_coordinates[i] for i in range(NN)]
zone_min_length = min(zone_lengths)
N = np.array([int(N_min * zone_lengths[i] / zone_min_length) for i in range(NN)])

# Генерация сетки
x = np.linspace(zone_coordinates[0], zone_coordinates[1], N[0] + 1)

for i in range(1, NN):
    x = np.hstack([x[:-1], np.linspace(zone_coordinates[i], zone_coordinates[i + 1], N[i] + 1)])
xs = x.size

# Генерация значений f, type_x и h
f = np.array([zone_f[define_zone_num(coord)] for coord in x])
type_x = np.array(list(map(define_type_x, x)))
h = np.array([x[i + 1] - x[i] for i in range(xs - 1)])

# Генерация системы
p_u = np.zeros(xs * 2)
# Уравнения для p внутри зон → xs - len(zone_coordinates)
# Уравнения для u внутри зон → xs - len(zone_coordinates)
# Граничные условия для p → 2
# Граничные условия для u → 2
# Равенства u в узлах зон → len(zone_coordinates) - 2
# Равенства p в узлах зон → len(zone_coordinates) - 2

A = np.zeros((xs * 2, xs * 2))
b = np.zeros(xs * 2)
ec = 0  # equation counter

for i in range(xs):
    if type_x[i] == 'inside':
        A[ec][i - 1] = 1
        A[ec][i + 1] = 1
        A[ec][i] = -2
        A[ec + xs - len(zone_coordinates)][xs + i] = 1
        A[ec + xs - len(zone_coordinates)][i + 1] = f[i] / h[i]
        A[ec + xs - len(zone_coordinates)][i] = -f[i] / h[i]
        ec += 1

ec *= 2
A[ec][0] = 1
b[ec] = 1
ec += 1
A[ec][xs - 1] = 1
ec += 1

A[ec][xs] = 1
A[ec][0] = -f[0] / h[0]
A[ec][1] = f[0] / h[0]
ec += 1

A[ec][2 * xs - 1] = 1
A[ec][xs - 2] = -f[xs - 2] / h[xs - 2]
A[ec][xs - 1] = f[xs - 2] / h[xs - 2]
ec += 1

for i in range(xs):
    if type_x[i] == 'knot':
        A[ec][i - 1] = -f[i - 1] / h[i - 1]
        A[ec][i + 1] = -f[i] / h[i]
        A[ec][i] = - A[ec][i - 1] - A[ec][i + 1]
        A[ec + len(zone_coordinates) - 2][i - 1] = -1
        A[ec + len(zone_coordinates) - 2][i + 1] = 1
        A[ec + len(zone_coordinates) - 2][xs + i] = h[i] / f[i] + h[i - 1] / f[i - 1]

        ec += 1

# вывод результатов
p_u = np.linalg.solve(A, b)
p = p_u[:xs]
u = p_u[xs:]

p_an = np.zeros(xs)
u_an = np.zeros(xs)
if task_number == 1:
    p_an = [p_analit_1(x[i]) for i in range(xs)]
    u_an = [u_analit_1(x[i]) for i in range(xs)]
elif task_number == 2:
    p_an = [p_analit_2(x[i]) for i in range(xs)]
    u_an = [u_analit_2(x[i]) for i in range(xs)]
elif task_number == 3:
    p_an = [p_analit_3(x[i]) for i in range(xs)]
    u_an = [u_analit_3(x[i]) for i in range(xs)]

plt.figure(1)
plt.plot(x, p_an, 'b', x, p, 'r--')
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
plt.ylim([-0.1, 1.1])
name_p_file = f'Функция давления (задание {task_number}).png'
plt.savefig(name_p_file, dpi=300)

plt.figure(2)
plt.plot(x, u_an, 'b', x, u, 'r--')
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
plt.ylim([-0.1, 1.1])
name_u_file = f'Функция скорости фильтрации (задание {task_number}).png'
plt.savefig(name_u_file, dpi=300)
