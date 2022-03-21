import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from matplotlib import pyplot as plt
import math


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


plt.figure(1)
for R in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    print(R)
    u0_arr = list()
    for K in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(K, end=' ')
        N_min = 10000
        # Кол-во зон и их границы
        with open('zone_info(u(rw,K))).txt', 'r') as file:
            zone_coordinates = np.array(list(map(float, file.readline().split())))
            zone_f = np.array(list(map(float, file.readline().split())))
        zone_coordinates[0] /= R
        zone_f[1] = K
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
                data.extend([-1 - 2 * x[i] / h[i], 1 + x[i] / h[i], x[i] / h[i]])
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
        u = p_u[xs:]
        u0_arr.append(u[0])
    print()
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], u0_arr)
plt.legend(('R=0.01', 'R=0.02', 'R=0.05', 'R=0.1', 'R=0.2', 'R=0.5'))
plt.title('Зависимость дебита скважины от rw и K')
plt.minorticks_on()
plt.grid()
plt.xlabel('K')
plt.ylabel('q')
plt.minorticks_on()
plt.grid(which='major', linewidth=1)
plt.grid(which='minor', linestyle=':')
plt.grid(which='minor', linestyle=':')
name_file = f'Зависимость дебита скважины от rw и K.png'
plt.savefig(name_file, dpi=300)
