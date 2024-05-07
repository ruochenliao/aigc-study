import jieba

if __name__ == "__main__":
    sentence = "我爱自然语言"
    seg_list = jieba.cut_for_search(sentence)
    print("搜索引擎模式:" + " ".join(seg_list))
    seg_list = jieba.cut(sentence, False)
    print("精确模式:" + " ".join(seg_list))
    seg_list = jieba.cut(sentence, True)
    print("全模式:" + " ".join(seg_list))
