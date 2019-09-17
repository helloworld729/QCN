from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

dir = 'G:/pythonCode/thesis/light_QA/QCN-systerm_debug/model_value/'

"""
用验证或者测试集的结果 和 真实的数据比较，对模型的效果做评估
"""

def plot_matrix(in_matrix):
    plt.matshow(in_matrix)
    plt.title('confusion_matrix')
    plt.colorbar()
    plt.ylabel('true')
    plt.xlabel('predict')
    plt.show()


def get_label(filename):
    y_hat = []
    idList = []
    with open(dir + filename, 'r', encoding='utf-8') as f:
        for raw in f.readlines():
            [id, label] = raw.strip().split()
            idList.append(id)
            y_hat.append(label)
    return idList, y_hat


def model_evaluate(goldname, predictname):

    goldId, y = get_label(goldname)
    predictId, y_hat = get_label(predictname)

    if len(y) == len(y_hat):

        plot_matrix(confusion_matrix(y, y_hat))

        acc = metrics.precision_score(y, y_hat, average='macro')  # 微平均，精确率
        recall = metrics.recall_score(y, y_hat, average='micro')
        F1 = metrics.f1_score(y, y_hat, average='weighted')
        print('准确率：{}\n召回率：{}\nF1:{}'.format(acc, recall, F1))

    else:
        print('文件不等长')


if __name__ == '__main__':

    model_evaluate('gold', 'predict')







