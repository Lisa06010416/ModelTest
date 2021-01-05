## MLflow

* 有可視化界面
* 可以記錄參數檔案與模型參數，並在可是化界面裡查看
* 打包，先需要有anaconda與MLflow，會在anaconda開新的指定環境

教學 :

* [入門教學](https://medium.com/ai-academy-taiwan/mlflow-a-machine-learning-lifecycle-platform-%E5%85%A5%E9%96%80%E6%95%99%E5%AD%B8-5ec222abf5f8)

* [官方文件](https://mlflow.org/docs/latest/index.html)

* [在colab上執行](https://medium.com/swlh/hyperparameter-tuning-with-mlflow-tracking-b67ec4de18c9)
    * [colab連線到本機](https://research.google.com/colaboratory/local-runtimes.html)
    * mlflow 跟 jupyter 要在同一個目錄下
    * 連線到本機後，不能mount google drive   
    jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0

-----------------------------

## Kubeflow

* Google 開源
* 簡化Kubernetes的流程
* [初探基本功能與概念](https://k2r2bai.com/2018/03/15/kubernetes/kubeflow/quick-start/)



---------------------------

## Kubernetes

* 常簡稱為**K8s**

* 自動部署、擴充和管理「容器化（containerized）應用程式」的開源系統

* Google設計並捐贈給Cloud Native Computing

* 電子書 - [Kubernetes](https://www.oreilly.com/library/view/kubernetes/9781492048718/)

  
-------------------

## TensorFlow Extended(TFX)

* TensorFlow based
* 由一開始的資料驗證與分析 => 模型訓練與評估(也可以比較新模型是否比舊模型好) => Pusher (可以自動部屬到目標環境中工作)
*  [TensorFlow Lite](https://link.zhihu.com/?target=https%3A//tensorflow.google.cn/lite)、[TensorFlow JS](https://link.zhihu.com/?target=https%3A//tensorflow.google.cn/js)、[TensorFlow Serving](https://link.zhihu.com/?target=https%3A//tensorflow.google.cn/tfx/guide/serving)
* 可攜至多種環境和自動化調度管理架構( [Apache Airflow](https://tensorflow.google.cn/tfx/guide/airflow)、[Apache Beam](https://tensorflow.google.cn/tfx/guide/beam_orchestrator) 和 [Kubeflow](https://tensorflow.google.cn/tfx/guide/kubeflow))
* 攜至不同的運算平台，包括內部部署平台和 [Google Cloud Platform (GCP)](https://cloud.google.com/) 等雲端平台。特別的是，TFX 可與多個受管理的 GCP 服務互通，例如用於[訓練和預測](https://cloud.google.com/ml-engine/)的 [Cloud AI 平台](https://cloud.google.com/ai-platform/)，以及用於分散式資料處理的 [Cloud Dataflow](https://cloud.google.com/dataflow/) (適用於機器學習生命週期的其他多個層面)。



-------------------------

## Airflow

* [一段 Airflow 與資料工程的故事：談如何用 Python 追漫畫連載](https://leemeng.tw/a-story-about-airflow-and-data-engineering-using-how-to-use-python-to-catch-up-with-latest-comics-as-an-example.html)
* Airbnb 誕生並開源，以 Python 寫成的工作流程管理系統（Workflow Management System）
* 利用 [DAG](https://airflow.apache.org/concepts.html#dags) 一詞來代表一種特殊的工作流程（Workflow）。如工作流程一樣，DAG（**D**irected **A**cyclic **G**raph）定義了我們有什麼工作、工作之間的執行順序以及依賴關係。DAG 的最終目標是將S所有工作依照上下游關係全部執行，而不是關注個別的工作實際上是怎麼被實作的



-------
0.  跑看看上面套件的demo
0.  TensorBoard
0.  TensorFlow Serving

        
1. pytorch 訓練一個小模型，使用Cython跟sPacy
2. 使用MLflow做訓練管理
3. 用ONNX將模型轉到tensorflow
4. Kubeflow
5. TFX