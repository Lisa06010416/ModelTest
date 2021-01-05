from transformers import BertTokenizer

# 中文會斷成一個一個字
# 英文或是數字會用n-gram的方式斷開
# 英文要轉小寫不然會是unknown tag

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

s1 = "列举无声电影类型的电影。"
t1 = tokenizer(s1)
for t in t1["input_ids"]:
    print(tokenizer.decode(t))

s2 = "谁是X战警II：突变体的堕落的生产者"
t2 = tokenizer(s2)
for t in t2["input_ids"]:
    print(tokenizer.decode(t))

s3 = "字符在穿刺穿刺者：德古拉的真实故事中说什么语言"
t3 = tokenizer(s3)
for t in t3["input_ids"]:
    print(tokenizer.decode(t))

s4 = "是第2卷光盘或dvd"
t4 = tokenizer(s4)  
for t in t4["input_ids"]:
    print(tokenizer.decode(t))

s5 = "什么是16958克拉森分类为..."
t5 = tokenizer(s5)
for t in t5["input_ids"]:
    print(tokenizer.decode(t))

s6 = "哪个系列包含尼康coolpix l15"
t6 = tokenizer(s6)
for t in t6["input_ids"]:
    print(tokenizer.decode(t))

print("paddinf test")
s7 = [
    "哪个系列包含尼康coolpix l15",
    "哪个系列包含尼康coolpix l15"
]
t7 = tokenizer(s7, truncation=True, padding='max_length')
print(t7['input_ids'])
print(t7['token_type_ids'])
print(t7['attention_mask'])
print(t7)
