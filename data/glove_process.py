j = 0
with open ('glove.840B.300d.txt', 'r', encoding='utf-8') as f:
    embeding = list(map(float, f.readline().split()[1:]))
    print(embeding)
    print(len(embeding), type(embeding), type(embeding[2]))
