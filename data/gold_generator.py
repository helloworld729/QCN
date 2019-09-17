# coding=utf-8
# 数据集统计
import sys
import random
from nltk.tokenize import word_tokenize
from xml.etree.ElementTree import parse

dir = 'G:/pythonCode/thesis/light_QA/QCN-systerm/scorer/'


def strList2str(strList):
    return_str = ''
    for str in strList:
        return_str = return_str + str + ' '
    return return_str.strip()


def get_data(flist):
    nq = na = 0
    for fname in flist:
        doc = parse('xml/' + fname + '.xml')                    # 煲汤-->解析对象
        write_str = ''
        for t in doc.iterfind('Thread'):                        # Thread类
            # q = t.find('RelQuestion')                           # Thread类中的Question类
            nq += 1                                             # num of question 加1

            for c in t.iterfind('RelComment'):                  # Thread类中的answer类
                RELC_ID = c.attrib['RELC_ID']
                evaluate = c.attrib['RELC_RELEVANCE2RELQ']
                na += 1                                         # num of answer + 1

                write_str = RELC_ID + '\t' + evaluate

                path = dir + '/' + fname + '/' + 'gold'
                with open(path, "a", encoding='utf-8') as f:
                    f.write(write_str)
                    f.write('\n')


        print('# file_name: {} # 问题总数.{}\t# 答案总数.{}\t '.format(fname, nq, na))


if __name__ == '__main__':
    # get_data(['15dev', '15test'])
    # get_data(['16dev', '16test'])
    get_data(['15dev'])




