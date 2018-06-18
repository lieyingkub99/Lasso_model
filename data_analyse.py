import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#--计算MSE--
def MSE(true, pred):
    N = len(pred)
    mse = sum((true - pred) ** 2) / N
    print("==========两次的MSE是=========：", float(mse))
    return mse
    plt.show()
#--画趋势图--
def evaluate_plot(pred,reference):
    plt.plot([x for x in range(300)], pred[:300],label = "result_1")
    plt.plot([x for x in range(300)], reference[:300],label = "result_2")
#--画偏差图--
def evaluate_plot_err(pred):
    plt.plot([x for x in range(300)], pred[:300],label = "err")

#--画分布图--
def plot_curve(pred,reference):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.kdeplot(pred)
    sns.kdeplot(reference)
    plt.show()

if __name__ == '__main__':
    #--读数据--

    for i in range(1, 9):

        data = pd.read_csv("data/result_{}.csv".format(i), header=None)
        data.columns = ["label_{}".format(i), "pred_{}".format(i)]
        #--计算MSE-
        MSE(data["label_{}".format(i)], data["pred_{}".format(i)])
        #--画分布图--
        #plot_curve(data.label, data.pred)
        #--画趋势图--
        evaluate_plot(data["label_{}".format(i)], data["pred_{}".format(i)])
        #--画偏差图--
        evaluate_plot_err(data["label_{}".format(i)]-data["pred_{}".format(i)])
        plt.legend()
        plt.show()