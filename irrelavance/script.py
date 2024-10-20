import re
import jieba
from collections import Counter

def tokenize(text):
    # 英文和数字按单词分割，中文使用jieba分词
    words = []
    # 提取英文单词、数字和中文字符
    tokens = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', text)
    
    for token in tokens:
        # 对中文部分进行jieba分词
        if re.match(r'[\u4e00-\u9fff]+', token):
            words.extend(jieba.lcut(token))
        else:
            words.append(token)
    
    return words

def count_frequencies(text):
    # 分词
    tokens = tokenize(text)
    # 统计词频
    counter = Counter(tokens)
    # 按照出现频率排序
    sorted_counter = counter.most_common()
    return sorted_counter

def write_to_file(filename, result):
    # 将统计结果写入文件
    with open(filename, 'w', encoding='utf-8') as f:
        for word, freq in result:
            f.write(f"'{word}': {freq}\n")

# 示例文本，包含表情符号
with open('test.txt', 'r') as f:
    texts = f.readlines()
    
text = ''
for line in texts:
    text += line

# 统计高频词
result = count_frequencies(text)

# 输出结果到文件
output_file = "word_frequencies.txt"
write_to_file(output_file, result)

print(f"统计结果已保存到 {output_file}")
