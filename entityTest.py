import jieba.posseg as pseg

if __name__ == "__main__":
    text = "赵云是三国时候的著名将领，他带兵打战"
    words = pseg.cut(text)

    for word, flag in words:
        print(f"{word}:{flag}")