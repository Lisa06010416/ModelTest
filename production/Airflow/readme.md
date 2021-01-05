#### comic.py
* 實作[一段 Airflow 與資料工程的故事：談如何用 Python 追漫畫連載](https://leemeng.tw/a-story-about-airflow-and-data-engineering-using-how-to-use-python-to-catch-up-with-latest-comics-as-an-example.html) 的教學
* Airflow 安裝 in windows
    * [安裝wls以及Ubuntu](https://xenby.com/b/226-%E6%8E%A8%E8%96%A6-wsl-windows-subsystem-for-linux-%E5%AE%89%E8%A3%9D%E8%88%87%E4%BD%BF%E7%94%A8%E6%95%99%E5%AD%B8)
    * 安裝Airflow

        * 沒提到的照 : [in windows](https://medium.com/@cchangleo/airflow-%E6%96%B0%E6%89%8B%E5%BB%BA%E7%BD%AE%E6%B8%AC%E8%A9%A6-part-1-9e0fe4d12e7a)
        * 先執行 sudo apt-get update
        * mysql
            * pip install MySQL-python 改為安裝 pip install mysqlclient
            * 安裝完執行 sudo service mysql start
            * mysql start有問題 mkdir: cannot create directory ‘//.cache’: Permission denied，因為用sudo執行的畫會讀不到&HOME，目前先參考網路的方法，直接跳過
            ```
            執行 :
            sudo nano /etc/profile.d/wsl-integration.sh
            在第一行前面加上 :
            if [[ "${HOME}" == "/" ]]; then
            exit 0
            fi
            之後重啟cmd
            ```
        * pip3 install --upgrade pip setuptools
        * 安裝airflow那行用 : pip install apache-airflow
        * airflow install 完
        ```
        sudo musql
        set global explicit_defaults_for_timestamp=1; 
        
        nano /airflow/airglow.cfg
        修改DB
        sql_alchemy_conn = mysql://airflow:airflow@localhost:3306/airflow
        ```
    * [airflow 容器部屬](https://medium.com/@cchangleo/airflow-with-docker-%E5%AE%B9%E5%99%A8%E9%83%A8%E7%BD%B2-part2-8ddb83dc2d4a)
