
## [100 Times Faster Natural Language Processing in Python](https://medium.com/huggingface/100-times-faster-natural-language-processing-in-python-ee32033bdced)
看了標題那篇文章後來看看一些套件與功能

## cPython
* [Cython 安裝](https://jarvus.dragonbeef.net/note/noteCython.php)
* [profiling](https://cython.readthedocs.io/en/latest/src/tutorial/profiling_tutorial.html)
    * 加入 .coveragerc 檔案，內容為
    ```
    [run]
    plugins = Cython.Coverage
    ```
    * 執行
     ````
     cython  --annotate-coverage coverage.xml  file_name.pyx
     ````
    * 在pyx檔案上加
    ````
    # cython: linetrace=True
    # distutils: define_macros=CYTHON_TRACE=1
    ````


## pstats
* [sPacy + Keras](https://kknews.cc/zh-tw/tech/v9av24y.html)
* [spacy-transformers](https://spacy.io/universe/project/spacy-transformers)
* [有pretrain好的模型](httspacy-transformersps://spacy.io/models)
* displaCy - 可視化套件，可以畫出dependency以及POS/NER，還可以用在網頁上

## spacy-transformers 
* [github](https://github.com/explosion/spacy-transformers)
* 提供寫好的Spacy的架構讓我們使用HunggingFace的Transformers
* 


