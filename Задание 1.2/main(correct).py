import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import time


def p_analit(x):
    return 1 - 2 / 11 * x if x < 0.5 else 20 / 11 - 20 / 11 * x


def u_analit(x):
    return 2 / 11


def func(x: float) -> float:
    return 1.0 if 0 <= x < 0.5 else 0.1


Narr = list()
Eparr = list()
Euarr = list()
Farr = list()
for N in range(1,100,2):
    start = time.time()
    print(N)

    # Генерация сетки
    x = np.linspace(0, 1, N + 1)

    xs = x.size

    # Генерация значений f, type_x и h
    # f = np.array([func(x[i]) for i in range(xs)])

    h = np.array([x[i + 1] - x[i] for i in range(N)])

    M = 100
    f = np.array([M / sum([1 / func(x[i] + j * h[i] / M) for j in range(M)]) for i in range(N)])

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

    for i in range(1, xs - 1):
        A[ec][i - 1] = f[i - 1] / h[i - 1]
        A[ec][i + 1] = f[i] / h[i]
        A[ec][i] = -f[i - 1] / h[i - 1] - f[i] / h[i]
        ec += 1
        A[ec][xs + i] = 1
        A[ec][i + 1] = f[i] / h[i]
        A[ec][i] = -f[i] / h[i]
        ec += 1

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

    # вывод результатов
    p_u = np.linalg.solve(A, b)
    p = p_u[:xs]
    u = p_u[xs:]

    p_an = [p_analit(x[i]) for i in range(xs)]
    u_an = [u_analit(x[i]) for i in range(xs)]

    # Расчет невязки
    Ep = sqrt(sum([(p[i] - p_an[i]) ** 2 for i in range(xs)]) / xs)
    Eu = sqrt(sum([(u[i] - u_an[i]) ** 2 for i in range(xs)]) / xs)

    Narr.append(N)
    Eparr.append(Ep)
    Euarr.append(Eu)
    Farr.append(f[N // 2])
    # plt.figure(1)
    # plt.plot(x, p_an, 'b', x, p, 'r--')
    # plt.title('Функция давления')
    # plt.minorticks_on()
    # plt.grid()
    # plt.legend(('Точное решение', 'Численное решение'))
    # plt.xlabel('x')
    # plt.ylabel('p')
    # plt.minorticks_on()
    # plt.grid(which='major', linewidth=1)
    # plt.grid(which='minor', linestyle=':')
    # plt.grid(which='minor', linestyle=':')
    # plt.show()
    #
    # plt.figure(2)
    # plt.plot(x, u_an, 'b', x, u, 'r--')
    # plt.title('Функция скорости фильтрации')
    # plt.minorticks_on()
    # plt.grid()
    # plt.legend(('Точное решение', 'Численное решение'))
    # plt.xlabel('x')
    # plt.ylabel('u')
    # plt.minorticks_on()
    # plt.grid(which='major', linewidth=1)
    # plt.grid(which='minor', linestyle=':')
    # plt.grid(which='minor', linestyle=':')
    # plt.show()
    print(f'время: {time.time() - start}')
plt.figure(3)
plt.plot(Narr, Eparr, 'r')
plt.minorticks_on()
plt.grid()
plt.xlabel('N')
plt.ylabel('Ep')
plt.minorticks_on()
plt.grid(which='major', linewidth=1)
plt.grid(which='minor', linestyle=':')
plt.grid(which='minor', linestyle=':')
plt.show()

plt.figure(4)
plt.plot(Narr, Euarr, 'r')
plt.minorticks_on()
plt.grid()
plt.xlabel('N')
plt.ylabel('Eu')
plt.minorticks_on()
plt.grid(which='major', linewidth=1)
plt.grid(which='minor', linestyle=':')
plt.grid(which='minor', linestyle=':')
plt.show()

plt.figure(5)
plt.plot(Narr, Farr, 'r')
plt.minorticks_on()
plt.grid()
plt.xlabel('N')
plt.ylabel('F(problem)')
plt.minorticks_on()
plt.grid(which='major', linewidth=1)
plt.grid(which='minor', linestyle=':')
plt.grid(which='minor', linestyle=':')
plt.show()
