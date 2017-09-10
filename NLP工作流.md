## 一、文本预处理
文本是一种高度非结构化的数据，含有各种噪声，在进行文本分析之前需要对文本进行预处理。

> 文本预处理：对文本进行清洗和标准化，以减少噪声为文本分析做准备的过程。

预处理过程一般包括：

1. 去除噪声：去除不相关字符，如标点、无意义的冠词。这些词一般称作“停用词（stopword）”
2. 分词：
3. 词汇规范化：
    1. 词干提取（stemming）:将单复数、时态还原为最初的样子
    2. 词形还原（lemmatization）:
    3. 对象标准化    
4. 语法检测和拼写纠正

下图表示文本预处理的一般流程：

原始文本→移除噪声（停用词/urls）→分词→单词规范化（词干提取/词形还原）→单词标准化（缩写等）→干净的文本

![](/assets/1.png)

### 1.1 去除噪声
> 噪声：任何与文本数据内容不相关或与最终输出无关的文本

用于去噪的方法：

1. 通过噪声字典去除指定噪声：定义一个噪声字典，然后遍历文本中所有单词，去除那些包含在噪声字典中的单词。

```python
# 移除文本噪声的示范代码
noise_list = ["is", "a", "this", "..."]
def _remove_noise(input_text):
    words = input_text.split()
    noise_free_words = [word for word in words if word not in
    noise_list]
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text_remove_noise("this is a sample text")
>>> "sample text"
```

2. 使用正则表达式移除具有固定格式的噪声,详见[这里](https://www.analyticsvidhya.com/blog/2015/06/regular-expression-python/)：

```python
# 移除固定模式噪声的示例代码
import re 
def _remove_regex(input_text, regex_pattern):
    urls = re.finditer(regex_pattern, input_text)
    for i in urls:
        input_text = re.sub(i.group().strip(), '', input_text)
    return input_text
regex_pattern = "#[\w]*"
  
_remove_regex("remove this #hashtag from analytics vidhya", regex_pattern)
>>> "remove this  from analytics vidhya"
```

### 1.2 词汇规范化
> 词汇标准化：同一个词的不同表现形式，如"play", "player", "played", "plays", "playing"代表了同一个词，只是词性或时态不同，我们需要将所有这一类词规范化为统一格式。

规范化是文本数据特征工程的主要步骤，它把高维特征映射到低维空间，这对很多机器学习模型而言都是理想情况。

最常用的词汇规范化技术：

1. 词干提取：词干提取基于一些基本规则将词汇的后缀（"ing", "ly", "es", "s"等）剥离的过程
2. 词形还原：词形还原是一种逐步还原词根的过程,它基于词汇和词法分析实现这个目的。

```python
# python中的NLTK库实现词法分析和词干提取的示例代码
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

word = "multiplying"
lem.lemmatize(word, "v")
>> "multiply"
stem.stem(word)
>> "multipli"
```

### 1.3 对象标准化
> 对象标准化：文本数据中有些词或短语无法再标准字典中找到，因而无法被模型识别，需要将它们转化为能够在标准字典中找到的对象。

方法：通过正则表达式或手工收集的对照字典，进行替换

```python
lookup_dict = {'rt':'Retweet', 'dm':'direct message', "awsm" : "awesome", "luv" :"love", "..."}
def _lookup_words(input_text):
    words = input_text.split()
    new_words = []
    for word in words:
        if word.lower() in lookup_dict:
            word = lookup_dict[word.lower()]
        new_words.append(word)
        new_text = " ".join(new_words)
        return new_text

_lookup_words("RT this is a retweeted tweet by Shivam Bansal")
>> "Retweet this is a retweeted tweet by Shivam Bansal"

```

## 二、文本特征工程
为了分析预处理后的文本数据，我们需要将其转化为特征。基于不同的用途，可以选用不同的技术来构建文本特征：
1. 语法分析
2. 实体/N元文法/基于单词的特征
3. 统计特征
4. 单词嵌入

### 2.1 语法分析
语法分析包括对句中的单词进行语法和排列分析，以显示单词间的关系。依存文法和词性标记是文本语法的重要属性。

#### 依赖树（Dependency Trees）
句子是单词的集合，句子中单词间的相互关系由基本的依存文法决定。依存文法是用于处理两个单词(带标记)间非对称二元关系的语法分析方法。每种关系可以通过一个三元组(relation, governor, dependent)来表示。

例如：考虑句子“Bills on ports and immigration were submitted by Senator Brownback, Republican of Kansas.” 该句单词的关系可以用如下的树状结构表示：

![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/01/11181146/image-2.png)

单词“submitted”是这句话的根单词(root word)，并且其与两棵子树相连(主体子树和客体子树)。每个子树本身又是一棵依存树，例如：(“Bills” <-> “ports” <by> “proposition” relation), (“ports” <-> “immigration” <by> “conjugation” relation)。

当我们以自顶向下的方式对依存树进行递归解析时，可以将“语法关系三元组”作为输出，输出可作为许多nlp问题的特征，如情感分析、实体识别和文本分类等。python的第三方包StanfordCoreNLP（由斯坦福NLP小组提供，仅供商业使用）和NLTK库中的依存文法可以用来生成依存树。

#### 2.2 词性标记
除了语法关系，句中的每个单词也通过词性(名词、动词、形容词、副词)进行相互关联。词性标记定义了单词在句中的用途和功能。宾夕法尼亚大学提供了一个完整的词性标记列表。以下代码则使用了NLTK库来对输入的文本进行词性标注：

```python
from nltk import word_tokenize, pos_tag
text = "I am learning Natural Language Processing on Analytics Vidhya"
tokens = word_tokenize(text)
print pos_tag(tokens)
>>> [('I', 'PRP'), ('am', 'VBP'), ('learning', 'VBG'), ('Natural', 'NNP'),('Language', 'NNP'),
('Processing', 'NNP'), ('on', 'IN'), ('Analytics', 'NNP'),('Vidhya', 'NNP')]
```

在NLP中，词性标注有个很多重要用途：

- 词义消歧

一些词的不同用法代表不同的意思. 如下列两句:
   
I. “Please book my flight for Delhi”

II. “I am going to read this book in the flight”

“Book” 在这里代表不同的意义, 好在它在两句的词性也不同. 第一句“book”是的动词, 第二句中它是个名词。 (Lesk Algorithm也被用于类似目的)

- word-based特征强化

学习模型可以在使用单词作为特征时学习单词的不同上下文，但如果他们被标记了词性，就能提供更强的特征。例如：

句子 -“book my flight, I will read this book”

单词 – (“book”, 2), (“my”, 1), (“flight”, 1), (“I”, 1), (“will”, 1), (“read”, 1), (“this”, 1)

- 标准化与词形还原：词性标记是词形还原（把单词还原为基本形）的基础步骤之一，

- 有效去除停用词：词性标记在移除停用词中很有用

举个例子，有一些标注可以定义某些语言中的低频词和不重要的词，像方位介词，数量词和语气助词等。

### 2.3 实体抽取（实体作为特征）
> 实体：句子中最重要的组成部分，名词短语和动词短语

实体识别算法是基于规则解析、词典搜索、词性标记和依赖解析的一类模型集合。实体识别算法可被应用于会话机器人、内容分析和消费者行为分析。

![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/01/11181407/image-3.png)

主题模型和命名实体识别是NLP中实体识别的两大主流方法：

#### 命名实体识别(NER)

> 命名实体识别:识别人名、地名和公司名等命名实体的过程

句子 – Sergey Brin, the manager of Google Inc. is walking in the streets of New York.

命名实体 – ( “person” : “Sergey Brin” ), (“org” : “Google Inc.”), (“location” : “New York”)

一个典型的NER模型包括三部分：

1. 名词短语识别：这一步根据依赖分析和词性标注从文本中提取所有名词短语
2. 短语分类：这一步对提取出的名词短语进行分类（位置、人名等）。Google地图的API提供了很好的位置归类方法，而利用dbpedia和wikipedia的数据库可以用来识别公司名和人名。除此之外，也可以将从各种渠道收集的数据整合为查找表和字典用于分类。
3. 实体消歧：有时一个实体会被错误地分类，所以我们需要对分类结果加一层检验。使用知识图谱可以实现这一目的，使用较多的知识图谱包括Google知识图谱，IBM Watson和Wikipedia。

#### 主题模型
> 主题建模：主题建模是自动识别文本所属主题的过程，它以无监督的方式从语料库中的单词中导出隐藏的模式。

> 主题：主题被定义为语料库中以词组形式重复出现的模式。

“health”，“doctor”，“patient”，“hospital”的一个好的主题模型结果是健康；“farm”, “crops”, “wheat” 以农业为主题。

隐含狄利克雷分布(LDA)是最流行的主题建模技术，以下是在python中使用LDA实现主题建模的代码。如果需要LDA模型的详细解释和技术细节，可以参考这篇[文章](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/)。

```
doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father." 
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc_complete = [doc1, doc2, doc3]
doc_clean = [doc.split() for doc in doc_complete]

import gensim from gensim
import corpora

# Creating the term dictionary of our corpus, where every unique term is assigned an index.  
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above. 
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

# Results 
print(ldamodel.print_topics())

```
#### n元语法作为特征
> N个单词构成的组合称为N元文法

N元文法（N > 1）通常比单词（一元文法）包含更多信息。此外，两元文法被认为是最重要的特征，下列代码能从文本中生成二元文法。

``` 
def generate_ngrams(text, n):
    words = text.split()
    output = []  
    for i in range(len(words)-n+1):
        output.append(words[i:i+n])
    return output

>>> generate_ngrams('this is a sample text', 2)
# [['this', 'is'], ['is', 'a'], ['a', 'sample'], , ['sample', 'text']] 
```
### 2.4 统计特征
文本数据也可以使用本节中描述的几种技术直接量化：

- 词组频率-逆文档频率 (TF – IDF)

> TF-IDF是一种常用于信息检索问题的加权模型，基于单词在文档中出现的频率将文本文档转化为向量模型。

假设有一个由N个文档组成的数据集，在任一文档D中，TF和IDF被定义为：

1. 词组频率(TF) – 词组“t”的TF被定义为其在文档“D”中的出现次数

2. 逆文档频率(IDF) – 词组的IDF被定义为其数据集中文档总数和包含该词组文档树的比值的自然对数。

3. TF-IDF - TF-ID 公式给出了词组在语料库（文档列表）中的相对重要性的度量，公式如下：

![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/01/11181616/image-4.png)

```
# 该模型会建立一个词汇字典，再为每个单词分配一个编号。
# 输出中的每一行包括一个元组（i，j）及文档i中标号为j的词组的TF-IDF值。
from sklearn.feature_extraction.text import TfidfVectorizer
obj = TfidfVectorizer()
corpus = ['This is sample document.', 'another random document.', 'third sample document text']
X = obj.fit_transform(corpus)
print X
>>>
(0, 1) 0.345205016865
(0, 4) ... 0.444514311537
(2, 1) 0.345205016865
(2, 4) 0.444514311537
```

- 计数/密度/可读性特征
基于频率或者密度的特征也被广泛用于建模和分析，这类特征可能看起来很简单却往往能发挥很大的作用。这类特征有：词频，句频，标点频率和行业术语频率。而另一类包括音节频率，SMOG指数和易读性指标的可读性特征，请参看[Textstat库](https://github.com/shivam5992/textstat)的相关内容。

### 2.5 单词嵌入（文本向量）
> 单词嵌入：单词嵌入是把单词表示为向量的现代方法，其目的是通过保留语料库中的语境相似度将高维单词特征重新定义为低维向量。它们被广泛应用于深度学习模型中，如卷积神经网络（CNN）和循环神经网络（RN）

 [Word2Vec](https://code.google.com/archive/p/word2vec/) 和 [GloVe](https://nlp.stanford.edu/projects/glove/)是单词嵌入的常用库，这些模型把文本语料库作为输入，并把文本向量作为输出。
 Word2Vec模型是由预处理模型和两个分别叫做Continuous Bag of Words 和skip-gram的浅层神经网络模型组成。这些模型被广泛应用于所有其他的nlp问题中。
 
 它首先从训练语料库中构建出一个词汇表，然后再学习单词嵌入表示
 下列代码使用gensim库把准备单词嵌入作为向量：
 
 ```
 # 它们可以作为机器学习模型的向量化输入，
 # 可以利用余弦定理来衡量文本相似性，还可以做词云和被用于文本分类
 from gensim.models import Word2Vec
sentences = [['data', 'science'], ['vidhya', 'science', 'data', 'analytics'],['machine', 'learning'], ['deep', 'learning']]

# train the model on your corpus  
model = Word2Vec(sentences, min_count = 1)

print model.similarity('data', 'science')
>>> 0.11222489293

print model['learning']  
>>> array([ 0.00459356  0.00303564 -0.00467622  0.00209638, ...])
 ```
 
## 三、NLP主要任务
本部分会讨论NLP领域的不同应用场景和常见问题

### 3.1 文本分类
文本分类是NLP的典型问题，具体例子包括：垃圾邮件识别、新闻主题分类、情感分析和搜索引擎的网页排序。

> 文本分类：文本分类通常被定义为一种对固定范畴的文本对象（文档或句子）进行系统分类的技术。当数据量很大时，从组织、信息过滤和存储方面考虑，分类就显得尤为重要。

一个典型的NLP分类器包含两部分：（a）训练 （b）预测。如下图所示，输入的文本会先经过预处理过程再被提取特征，之后相应的机器学习模型就会接受特征并对新文本作出类别预测。

![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/01/11182015/image-5.png)

这里的代码就使用了朴素贝叶斯分类器（包含在blob库中）来实现文本分类过程：

```
from textblob.classifiers import NaiveBayesClassifier as NBC
from textblob import TextBlob
training_corpus = [
                   ('I am exhausted of this work.', 'Class_B'),
                   ("I can't cooperate with this", 'Class_B'),
                   ('He is my badest enemy!', 'Class_B'),
                   ('My management is poor.', 'Class_B'),
                   ('I love this burger.', 'Class_A'),
                   ('This is an brilliant place!', 'Class_A'),
                   ('I feel very good about these dates.', 'Class_A'),
                   ('This is my best work.', 'Class_A'),
                   ("What an awesome view", 'Class_A'),
                   ('I do not like this dish', 'Class_B')]

test_corpus = [
               ("I am not feeling well today.", 'Class_B'),
               ("I feel brilliant!", 'Class_A'),
               ('Gary is a friend of mine.', 'Class_A'),
               ("I can't believe I'm doing this.", 'Class_B'),
               ('The date was good.', 'Class_A'),
               ('I do not enjoy my job', 'Class_B')]

model = NBC(training_corpus)
print(model.classify("Their codes are amazing."))
>>> "Class_A"
print(model.classify("I don't like their computer."))
>>> "Class_B"
print(model.accuracy(test_corpus))
>>> 0.83
```

Scikit-Learn也提供了文本分类的工作流程：

```
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics import classification_report
from sklearn import svm 
# 为SVM准备数据 (和朴素贝叶斯分类器使用同样特征)
train_data = []
train_labels = []
for row in training_corpus:
    train_data.append(row[0])
    train_labels.append(row[1])

test_data = []
test_labels = []
for row in test_corpus:
    test_data.append(row[0])
    test_labels.append(row[1])

# 创建特征向量
vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)
# 训练特征向量
train_vectors = vectorizer.fit_transform(train_data)
# 在测试集上应用模型 
test_vectors = vectorizer.transform(test_data)

# 训练使用线性核的SVM模型
model = svm.SVC(kernel='linear')
model.fit(train_vectors, train_labels)
prediction = model.predict(test_vectors)
>>> ['Class_A' 'Class_A' 'Class_B' 'Class_B' 'Class_A' 'Class_A']

print (classification_report(test_labels, prediction))
```
文本分类严重依赖于特征的质量和数量，当应用任何一种机器学习模型时，包含尽可能多的训练数据总是一种好的办法

### 3.2 文本匹配/相似度

NLP的另一重要领域是匹配文本对象来计算相似度。它可以应用在拼写纠正，数据去重和基因组分析等领域。

根据需求的不同有很多匹配技术可供选择，这里会详细介绍几种重要的技术：

#### Levenshtein 距离
> Levenshtein 距离：指两个字串之间，由一个转成另一个所需的最少编辑操作次数。许可的编辑操作包括将一个字符替换成另一个字符，插入一个字符，删除一个字符。下列代码使用一种很省内存的算法计算该距离

```python
def levenshtein(s1,s2): 
    if len(s1) > len(s2):
        s1,s2 = s2,s1 
    distances = range(len(s1) + 1) 
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1]) 
            else:
                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
        distances = newDistances 
    return distances[-1]

print(levenshtein("analyze","analyse"))
```
#### 语音匹配
语音匹配算法会把关键词（人名、地名等）作为输入，产生一个字符串，用于标识（粗略地）语音相似的一组单词。在对大型语料库进行索引，纠正拼写错误和匹配相关姓名时很有作用。Soundex和Metaphone是这一领域的两大主要算法，Python中的Fuzzy库用于计算不同单词的soundex字符串，比如：

```
import fuzzy 
soundex = fuzzy.Soundex(4) 
print soundex('ankit')
>>> “A523”
print soundex('aunkit')
>>> “A523” 
```

#### 柔性字符串匹配
完整的文本匹配系统包括流水线化的不同算法来计算各种文本变体。正则表达式对此也能发挥很大的作用，而其它一些常用技术则包括 - 精确字符串匹配，词形还原匹配和紧凑匹配等

#### 余弦相似性
余弦相似性通过测量两个矢量的夹角的余弦值来度量它们之间的相似性。以下代码把文本按照词频转换成向量并使用余弦值来衡量两个文本的相似度

```
import math
from collections import Counter
def get_cosine(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in common])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()]) 
    sum2 = sum([vec2[x]**2 for x in vec2.keys()]) 
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
   
    if not denominator:
        return 0.0 
    else:
        return float(numerator) / denominator

def text_to_vector(text): 
    words = text.split() 
    return Counter(words)

text1 = 'This is an article on analytics vidhya' 
text2 = 'article on analytics vidhya is about natural language processing'

vector1 = text_to_vector(text1) 
vector2 = text_to_vector(text2) 
cosine = get_cosine(vector1, vector2)
>>> 0.62 
```

### 3.3 共指消解（Coreference Resolution）
共指消解是一个寻找句内单词链接关系的过程。考虑如下例句：

“ Donald went to John’s office to see the new table. He looked at it for an hour.”

人类可以识别句中的“he”指代Donald而非John，“it”指代table而非office。NLP中的指代消解则可以自动实现这一过程。它被应用于文档总结，自动答题和信息提取等方面，斯坦福的CoreNLP项目提供了Python包供商业使用。

### 3.4 其他NLP任务
- 文本总结：给定文档或者段落，识别出其中的关键句。

- 机器翻译：自动翻译输入的人类语言，对语法、语义和信息保真度较高。

- 自然语言的生成和理解：把计算机数据库中的信息转化为可读的人类语言称为语言生成。而把文本块转化成计算机更容易处理的逻辑结构的过程称为语言理解。

- 光学字符识别（OCR）：给定图片，识别其中文字。

- 文档信息化：把文本数据（网页，文件，pdf和图像）中的信息转化为可信息的干净格式。

## 四、重要的NLP库
Scikit-learn: 机器学习库

Natural Language Toolkit (NLTK): 为各种NLP技术提供轮子

Pattern – 网页挖掘模块，和NLTK搭配使用

TextBlob – 方便的NLP工具的API，基于NLTK和Pattern架构

spaCy – 工业级的NLP库

Gensim – 可构建主题模型

Stanford Core NLP – 斯坦福NLP小组提供的各类服务