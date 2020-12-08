
## [100 Times Faster Natural Language Processing in Python](https://medium.com/huggingface/100-times-faster-natural-language-processing-in-python-ee32033bdced)
* [demo code](https://github.com/huggingface/100-times-faster-nlp/blob/master/100-times-faster-nlp-in-python.ipynb)

#### Cython
* 迴圈用Cython加速

* 但NLP中會涉及很多String的操作，可是Cython有警語說 :
    ```
     unless you know what you are doing, avoid using C strings where possible and use Python string objects instead.
    ```
    
* 因此需要用spaCy來處理String

* Cython with Jupyter :

    * 加入Cython擴充 : 

        ```
        %load_ext Cython
        ```

    * 定義Cython Cell

        ```
        %%cython
        ```

        



#### [spaCy](https://spacy.io/)

* Production導向

  

* [spaCy 學習 第一篇：核心型別](https://www.itread01.com/content/1557454262.html) 

  * 有4種不同的Object :

    * Doc - List of Token，一段文字處理後的結果

    * Span - A slice of Doc

    * Token - 一個Token被處理完後的資訊(包含原本來自哪個doc、在原本doc中的index、本身的text、POS的tag、...)

    * Vocab - 字典

      * 會使用到以下物件

        * Lexeme - Vocab 中的一個詞的物件
        * StringStore - 儲存64bit的hashvalue以及其對應的詞彙的字典物件，可以用hashvalue來查察尋詞，或是用詞來查詢對應的hashvalue

      * 屬性

        * string - Vocab 的 string本身是一個StringStore 的object

      * 初始化時傳入list of string，或是StringStore 來建立Vocab 

        

* 當我們使用nlp method時，會將text經過一連串的處理後產生Doc  object，可以自己定義要什麼樣的pipline以及自訂custom function

  ![spaCy_pipline](.\readme.assets\spaCy_pipline.PNG)

  

  
