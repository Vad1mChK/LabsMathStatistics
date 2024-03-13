"""
Выберите распределение, у которого существуют первые четыре момента, и экспериментально убедитесь в асимптотической
нормальности выборочного среднего, выборочной дисперсии, выборочной квантили порядка 0,5 для данного распределения.

Также экспериментально убедитесь В ТОМ, ЧТО 𝑛𝐹(𝑋(2)) → 𝑈1 ∼ Γ(2, 1) and 𝑛(1 − 𝐹(𝑋(𝑛))) → 𝑈2 ∼ Γ(1, 1) = Exp(1).

Указание: сгенерируйте достаточно большое количество выборок достаточно большого объема, для каждой сгенерированной
 выборки вычислите соответствующие статистики (функции от выборок), постройте гистограммы результатов для каждой
  статистики, для наглядности рядом е гистограммой можно нарисовать соответствующую плотность (пока это метод
   "на глаз", но в дальнейшем мы разберем статистическую процедуру, позволяющую проверить согласованность распределения
    выборки е заданным вероятностным законом), также можно помимо гистограммы вывести мат, ожидание, дисперсию (или
     стандартное отклонение) и медиану.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, expon


def plot_histogram(location, data, title: str, color: str, density: bool = True):
    location.hist(data, bins=25, alpha=0.6, density=density, color=color)
    location.set_title(title)


def plot_density(location, func, x_range, color: str = 'k'):
    y = func(x_range)
    location.plot(x_range, y, color=color, linewidth=2)


if __name__ == '__main__':
    n_samples = 10000
    sample_size = 1000

    a = 0
    b = 100

    samples = np.random.uniform(low=a, high=b, size=(n_samples, sample_size))

    sample_means = samples.mean(axis=1)
    sample_variances = samples.var(axis=1, ddof=1)
    sample_medians = np.median(samples, axis=1)

    sorted_samples = np.sort(samples, axis=1)

    u1 = n_samples * (sorted_samples[:, 1] - a) / (b - a)
    u2 = n_samples * (1 - (sorted_samples[:, -1] - a) / (b - a))

    fig, axs = plt.subplots(3, 2, figsize=(16.5 / 2.54, 25 / 2.54))

    plot_histogram(axs[0, 0], sample_means, title='Выборочные средние', color='r')
    means_mean = np.mean(sample_means)
    means_stddev = np.std(sample_means, ddof=1)
    xval = np.linspace(
        means_mean - 3 * means_stddev, means_mean + 3 * means_stddev, 1000
    )
    plot_density(axs[0, 0], lambda x: norm.pdf(x, means_mean, means_stddev), xval)

    plot_histogram(axs[1, 0], sample_variances, title='Выборочные дисперсии', color='g')
    variances_mean = np.mean(sample_variances)
    variances_stddev = np.std(sample_variances, ddof=1)
    xval = np.linspace(
        variances_mean - 3 * variances_stddev, variances_mean + 3 * variances_stddev, 1000
    )
    plot_density(axs[1, 0], lambda x: norm.pdf(x, variances_mean, variances_stddev), xval)

    plot_histogram(axs[2, 0], sample_medians, title='Выборочные медианы', color='m')
    medians_mean = np.mean(sample_medians)
    medians_stddev = np.std(sample_medians, ddof=1)
    xval = np.linspace(
        medians_mean - 3 * medians_stddev, medians_mean + 3 * medians_stddev, 1000
    )
    plot_density(axs[2, 0], lambda x: norm.pdf(x, medians_mean, medians_stddev), xval)

    plot_histogram(axs[0, 1], u1, title='Распределение U_1', color='y')
    xval = np.linspace(
        u1.min(), u1.max(), 1000
    )
    rv = gamma(a=2, scale=np.sqrt(u2.max() - u2.min()))
    plot_density(axs[0, 1], lambda x: rv.pdf(x), xval)

    plot_histogram(axs[1, 1], u2, title='Распределение U_2', color='c')
    xval = np.linspace(
        u1.min(), u2.max(), 1000
    )
    rv = expon(scale=np.sqrt(u2.max() - u2.min()))
    plot_density(axs[1, 1], lambda x: rv.pdf(x), xval)

    plt.tight_layout()
    plt.show()
    plt.savefig("Task 1")