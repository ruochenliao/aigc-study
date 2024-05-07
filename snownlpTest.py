from snownlp import SnowNLP
import jieba
if __name__ == "__main__":
    text = "我爱猫"
    words = jieba.cut(text)
    segement_text = " ".join(words)
    s = SnowNLP(segement_text)
    score = s.sentiments
    print(f"positive得分: {score}")