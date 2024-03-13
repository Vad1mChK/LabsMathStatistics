"""
В файле mobile_phones.csv приведены данные о мобильных телефонах. В сколько моделей можно вставить 2 сим-карты,
сколько поддерживают 3G, каково наибольшее число ядер У процессора? Рассчитайте выборочное среднее,
выборочную дисперсию, выборочную медиану и выборочную квантиль порядка 2/5, построить график эмпирической
функции распределения, гистограмму и bох-рlоt для емкости аккумулятора для всей совокупности и в отдельности
для поддерживающих/не поддерживающих Wi-Fi.
"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_edf(data: pd.Series, column_name: str, plot_name: str, save: bool = False):
    """
    Plots the EDF (empirical distribution function) of the given series.
    :param data: Series to plot.
    :param column_name: Column name (aka name of X axis).
    :param plot_name: Name of plot.
    :param save: Whether to save the plot to a file.
    :return: None
    """
    plt.ecdf(data, label='EDF')
    plt.xlabel(column_name)
    plt.ylabel('EDF')
    plt.title(plot_name)
    plt.grid(True)
    plt.legend()
    if save:
        plt.savefig(plot_name)
    plt.show()


def plot_hist(data: pd.Series, plot_name: str, save: bool = False):
    """
    Plots a histogram of the given series.
    :param data: Series to plot.
    :param column_name: Column name (aka name of X axis).
    :param plot_name: Name of plot.
    :param save: Whether to save the plot to a file.
    :return: None
    """
    plt.hist(data, bins=15, histtype='step')
    plt.title(plot_name)
    if save:
        plt.savefig(plot_name)
    plt.show()


def plot_boxplot(data: pd.Series, plot_name: str, save: bool = False):
    """
    Plots a boxplot of the given series.
    :param data: Series to plot
    :param plot_name: Name of plot.
    :param save: Whether to save the plot to a file.
    :return: None
    """
    plt.boxplot(data)
    plt.title(plot_name)
    plt.grid(True)
    if save:
        plt.savefig(plot_name)
    plt.show()


def plot_boxplot_df(data: pd.DataFrame, plot_name: str, save: bool = False):
    """
    Plots a boxplot of the given data frame.
    :param data: Dataframe to plot.
    :param plot_name: Name of plot.
    :param save: Whether to save the plot to a file.
    :return: None.
    """

    plt.figure(figsize=(8, 6))
    data.boxplot()
    plt.title(plot_name)
    plt.grid(True)
    if save:
        plt.savefig(plot_name)
    plt.show()


def main():
    df = pd.read_csv('mobile_phones.csv')

    # Count phones with dual SIM, phones with 3G, find max cores count
    print("Support for dual SIM:", df['dual_sim'].value_counts()[1])
    print("Support for 3G:", df['three_g'].value_counts()[1])
    print("Max cores count:", int(df.max()['n_cores']))
    print()

    df_wifi = df[df.apply(lambda row: row['wifi'] != 0, axis=1)]
    df_nowifi = df[df.apply(lambda row: row['wifi'] == 0, axis=1)]

    # Find mean of battery power
    print('Mean of battery power:', df.mean(axis=0)['battery_power'])
    print('Mean of battery power (with Wi-Fi):', df_wifi.mean(axis=0)['battery_power'])
    print('Mean of battery power (without Wi-Fi):', df_nowifi.mean(axis=0)['battery_power'])
    print()

    # Find variance of battery power
    print(
        'Variance of battery power:', df['battery_power'].var(ddof=0), '(biased),',
        df['battery_power'].var(), '(unbiased)'
    )
    print(
        'Variance of battery power (with Wi-Fi):', df_wifi['battery_power'].var(ddof=0), '(biased),',
        df_wifi['battery_power'].var(), '(unbiased)'
    )
    print(
        'Variance of battery power (without Wi-Fi):', df_nowifi['battery_power'].var(ddof=0), '(biased),',
        df_nowifi['battery_power'].var(), '(unbiased)'
    )
    print()

    # Find median of battery power
    print('Median of battery power:', df.median(axis=0)['battery_power'])
    print('Median of battery power (with Wi-Fi):', df_wifi.median(axis=0)['battery_power'])
    print('Median of battery power (without Wi-Fi):', df_nowifi.median(axis=0)['battery_power'])
    print()

    # Find 2/5 - quantile of battery power
    q = 2 / 5
    print(f'Quantile {q} of battery power:', df['battery_power'].quantile(q))
    print(f'Quantile {q} of battery power (with Wi-Fi):', df_wifi['battery_power'].quantile(q))
    print(f'Quantile {q} of battery power (without Wi-Fi):', df_nowifi['battery_power'].quantile(q))

    bp = df['battery_power']
    bp_wifi = df_wifi['battery_power']
    bp_nowifi = df_nowifi['battery_power']

    # Plot the empirical distribution function of battery power
    plot_edf(bp, 'Battery power', 'EDF of battery power (all)', save=True)
    plot_edf(bp_wifi, 'Battery power', 'EDF of battery power (with Wi-Fi)', save=True)
    plot_edf(bp_nowifi, 'Battery power', 'EDF of battery power (without Wi-Fi)', save=True)

    # Plot the histogram of battery power
    plot_hist(bp, 'Histogram of battery power (all)', save=True)
    plot_hist(bp_wifi, 'Histogram of battery power (with Wi-Fi)', save=True)
    plot_hist(bp_nowifi, 'Histogram of battery power (without Wi-Fi)', save=True)

    # Plot the boxplots of battery power
    plot_boxplot(bp, 'Box plot of battery power (all)', save=True)
    plot_boxplot(bp_wifi, 'Box plot of battery power (with Wi-Fi)', save=True)
    plot_boxplot(bp_nowifi, 'Box plot of battery power (without Wi-Fi)', save=True)

    plot_boxplot_df(pd.DataFrame(
        {
            'All': bp,
            'With Wi-Fi': bp_wifi,
            'Without Wi-Fi': bp_nowifi
        }
    ), 'Box plot of battery power (all, with Wi-Fi, without Wi-Fi)', save=True)


if __name__ == '__main__':
    main()
