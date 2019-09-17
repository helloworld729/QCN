# coding=utf-8
# 数据集统计
import sys
import random 
from nltk.tokenize import word_tokenize
from xml.etree.ElementTree import parse

dir = 'G:/pythonCode/thesis/light_QA/QCN-systerm_debug/data/xml/'


def strList2str(strList):
    return_str = ''
    for str in strList:
        return_str = return_str + str + ' '
    return return_str.strip()


def get_data(flist):
    nq = na = ls = lb = la = 0
    for fname in flist:
        doc = parse('xml/' + fname + '.xml')                    # 煲汤-->解析对象
        write_str = ''
        for t in doc.iterfind('Thread'):                        # Thread类
            q = t.find('RelQuestion')                           # Thread类中的Question类
            nq += 1                                             # num of question 加1
            subject = word_tokenize(q.findtext('RelQSubject'))  # 解析Question类中的subject
            body = word_tokenize(q.findtext('RelQBody'))        # 解析Question类中的body
            ls += len(subject)                                  # 主题总长
            lb += len(body)                                     # body总长

            subject = strList2str(subject)
            body = strList2str(body)

            for c in t.iterfind('RelComment'):                  # Thread类中的answer类
                RELC_ID = c.attrib['RELC_ID']
                evaluate = c.attrib['RELC_RELEVANCE2RELQ']
                answer = word_tokenize(c.findtext('RelCText'))  # 解析答案内容
                la += len(answer)                               # 答案总长加
                na += 1                                         # num of answer + 1

                answer = strList2str(answer)

                write_str = RELC_ID + '\t' + evaluate + '\t' + subject + '\t' + body + '\t' + answer

                # if  not 'train' in fname:
                #     write_str = RELC_ID + '\t' + write_str

                path = dir + fname + '.txt'
                with open(path, "a", encoding='utf-8') as f:
                    f.write(write_str)
                    f.write('\n')

        print('# file_name: {} # 问题总数.{}\t# 答案总数.{}\t 平均主题词长度.{}\t平均body长度.{}\tanswer的平均长度.{}'.format(fname, nq, na, ls/nq, lb/nq, la/na))


if __name__ == '__main__':
    get_data(['15train'])
    # get_data(['16train1', '16train2'])
    # get_data(['15train', '15dev', '15test'])
    # get_data(['16train1', '16train2', '16dev', '16test'])



