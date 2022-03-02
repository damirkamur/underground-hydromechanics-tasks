import numpy as np
from matplotlib import pyplot as plt
from math import sqrt


def p_analit(x):
    return 1 - x


def u_analit(x):
    return 1


# Тип сетки
# Если True → задать N
grid_regular = False

if grid_regular:
    N = 10
    x = np.linspace(0, 1, N + 1)
else:
    with open('grid.txt', 'r') as file:
        x = np.array(list(map(float, file.readline().split())))
        N = len(x) - 1

u = np.zeros(N + 1)
p = np.zeros(N + 1)
h = np.array([x[i + 1] - x[i] for i in range(N)])

# Заполнение матрицы и решение уравнений для P
A = np.zeros((N + 1, N + 1))
b = np.zeros(N + 1)

# Уравнения
for i in range(1, N):
    A[i][i - 1] = 1 / h[i - 1]
    A[i][i + 1] = 1 / h[i]
    A[i][i] = - A[i][i - 1] - A[i][i + 1]
# граничные условия
A[0][0] = 1
b[0] = 1
A[N][N] = 1
p = np.linalg.solve(A, b)

# Заполнение матрицы и решение уравнений для U
u[0] = -(p[1] - p[0]) / h[0]
for i in range(1, N):
    u[i] = -(p[i + 1] - p[i - 1]) / (h[i] + h[i - 1])
u[N] = -(p[N] - p[N - 1]) / h[N - 1]

# Расчет невязки
E = sqrt(sum([(p[i] - p_analit(x[i])) ** 2 for i in range(N + 1)]) / (N + 1))

# Графики
p_an = [p_analit(x[i]) for i in range(N + 1)]
u_an = [u_analit(x[i]) for i in range(N + 1)]

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
plt.savefig('Функция давления.png', dpi=300)

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
plt.savefig('Функция скорости фильтрации.png', dpi=300)

# Перерасчет на реальные величины
L, k, m, dp, mu = 100, 10e-12, 0.2, 10e6, 10e-3
u0 = k * dp / mu / L
u_dim = np.array(list(map(lambda x: x * u0, u)))
u_real = np.array(list(map(lambda x: x / m, u_dim)))
p1 = p[0] * dp
p2 = p[N] * dp
T = m * mu / k * L ** 2 / (p1 - p2)

with open('results.txt', 'w', encoding='utf-8') as file:
    file.write(f'Невязка при расчете давления: {E}\n')
    file.write(f'Скорость фильтрации в узлах (безразмерная): {u}\n')
    file.write(f'Скорость фильтрации в узлах (размерная): {u_dim}\n')
    file.write(f'Истинная скорость фильтрации в узлах (размерная): {u_real}\n')
    file.write(f'Время прохождения частицы между галереями: {T}')