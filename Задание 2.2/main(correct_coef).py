import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from matplotlib import pyplot as plt
import math

task_number = 1


def p_analit_1(x):
    global rw
    return 1 - math.log(x) / math.log(rw)


def u_analit_1(x):
    global rw
    return 1 / math.log(rw) / x


def p_ind(i: int) -> int:
    return i


def u_ind(i: int) -> int:
    global xs
    return xs + i


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
N_min = 10000

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
rw = x[0]

# Генерация значений f, type_x и h
f = np.array([zone_f[define_zone_num(coord)] for coord in x])
type_x = np.array(list(map(define_type_x, x)))
h = np.array([x[i + 1] - x[i] for i in range(xs - 1)])
tetta = np.array([-h[i] / x[i] / math.log(x[i] / x[i + 1]) for i in range(xs - 1)])

# Массивы для спарс матрицы
row_ind = list()
col_ind = list()
data = list()
rhs = np.zeros(2 * xs)
eq_num = 0

# Уравнения для внутренних узлов
for i in range(xs):
    if type_x[i] == 'inside':
        col_ind.extend([p_ind(i), p_ind(i + 1), u_ind(i)])
        row_ind.extend([eq_num, eq_num, eq_num])
        data.extend([-f[i] / h[i], f[i] / h[i], 1])
        eq_num += 1
        col_ind.extend([p_ind(i), p_ind(i + 1), p_ind(i - 1)])
        row_ind.extend([eq_num, eq_num, eq_num])
        data.extend([(x[i] + x[i + 1]) * tetta[i] + (x[i] + x[i - 1]) * tetta[i - 1], -(x[i] + x[i + 1]) * tetta[i],
                     -(x[i] + x[i - 1]) * tetta[i - 1]])
        eq_num += 1

# Граничные условия для p
col_ind.append(p_ind(0))
row_ind.append(eq_num)
data.append(1)
eq_num += 1
col_ind.append(p_ind(xs - 1))
row_ind.append(eq_num)
data.append(1)
rhs[eq_num] = 1
eq_num += 1
# Граничные условия для u
col_ind.extend([p_ind(0), p_ind(1), u_ind(0)])
row_ind.extend([eq_num, eq_num, eq_num])
data.extend([f[0] / h[0], -f[0] / h[0], -1])
eq_num += 1
col_ind.extend([p_ind(xs - 2), p_ind(xs - 1), u_ind(xs - 1)])
row_ind.extend([eq_num, eq_num, eq_num])
data.extend([f[xs - 2] / h[xs - 2], -f[xs - 2] / h[xs - 2], -1])
eq_num += 1

for i in range(xs):
    if type_x[i] == 'knot':
        col_ind.extend([p_ind(i - 1), p_ind(i + 1), p_ind(i)])
        row_ind.extend([eq_num, eq_num, eq_num])
        data.extend([-f[i - 1] / h[i - 1], - f[i] / h[i], f[i - 1] / h[i - 1] + f[i] / h[i]])
        eq_num += 1
        col_ind.extend([p_ind(i - 1), p_ind(i + 1), u_ind(i)])
        row_ind.extend([eq_num, eq_num, eq_num])
        data.extend([-1, 1, h[i] / f[i] + h[i - 1] / f[i - 1]])
        eq_num += 1

sA = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(2 * xs, 2 * xs))
# вывод результатов
p_u = linalg.spsolve(sA, rhs)
p = p_u[:xs]
u = p_u[xs:]

p_an = [p_analit_1(x[i]) for i in range(xs)]
u_an = [u_analit_1(x[i]) for i in range(xs)]

# Расчет невязки
E = math.sqrt(sum([(p[i] - p_an[i]) ** 2 for i in range(xs)]) / xs)

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
name_p_file = f'Функция давления (задание 1 п.к.) N={xs}.png'

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
name_u_file = f'Функция скорости фильтрации (задание 1 п.к.) N={xs}.png'

plt.savefig(name_u_file, dpi=300)

print(E)