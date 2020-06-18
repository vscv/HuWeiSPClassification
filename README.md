# HuWeiSPClassification

Check and send new images to tf inception v3 model that train on HuWei dataset.

# 使用方法
## 執行辨識
僅需在伺服器端輸入下列指令

    $ sudo docker run -v /$HOST/TF_io:/home/TF_io lswdokcer/tf_docker_v3

* $HOST：請將$HOST替換成您主機上的資料夾路徑
* TF_io: 固定名稱，存放新收影像檔與辨識結果
* lswdokcer: 為docker hub存放空間之名稱(若客製化後docker請改成您個人的空間)
* tf_docker_v3: v[] 為版本號


***
***

# 客製化需求
若有客製化需求才參考以下步驟。

## 安裝Docker環境
DOCKER Env

    $ sudo apt-get update
    $ sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
    $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    $ sudo apt-key fingerprint 0EBFCD88
    $ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs)  stable"

INSTALL DOCKER

    $ sudo apt-get update
    $ sudo apt-get install docker-ce (for Docker CE)

Check version of docker by:

    $ apt-cache madison docker-ce

If you want same version : 

    $sudo apt-get install docker-ce=<VERSION>

Verify that Docker CE : 
    
    $ sudo docker run hello-world

How show all containers on the system: 

    $ sudo docker ps -a

To stop a container:

    $ sudo docker stop $Name
    
To show only running containers use:

    $ docker ps 
    
To show all containers use:

    $ docker ps –a
    
To stop/start docker service: 

    $ sudo service docker stop

Lists all the images on your local system:

    $ docker images


## 修改Docker image

### TF_io
1. 若host主機已安裝好docker engine，另外還需建立輸出影像與輸出結果的資料夾。例如：host主機帳號下建立/host/name/$TF_io_root_dir/TF_io主資料夾，並在其下建立/in與/out資料夾。

2. 使用前請先確認每個鏡頭的ROI設定檔(00xxxxxlive.cfg)已放入host/name/$TF_io_root_dir/主資料夾中。當container 運行時，會自動將新加入TF_io/in/中的影像進行辨識，其中水域偵測需要依據每個鏡頭的ROI設定檔(00xxxxxlive.cfg)來進行處理，最後才輸出至TF_io/out/中。

$TF_io/主資料夾內容結構如下：

    in/
    out/
    001140live.cfg
    002051live.cfg
    009052live.cfg
    ...

### TF_run
存放模型與相關檔案，結構如下:

    TF_run/
       hw_confg/
       hw_model/
       Check.exe
       classify.exe
       RG.exe
     
## 功能增加與修改
請至/SRC/中修改個別的Py檔案，再使用pyinstaller封裝，並取代原本的TF_run/*.exe。

    $ pyinstaller -D -F -n Check.exe -c Check.py
    $ pyinstaller -D -F -n classify.exe -c classify.py 
    $ pyinstaller -D -F -n RG.exe -c RG.py
    $ pyinstaller -D -F -n RG_GBIS.exe -c RG_GBIS.py
    $ mv *.exe TF_run/

## 更新模型
請在Tensorflow 1.13環境重新訓練Inception v3，並將output_graph_HW.pb取代/TF_run/how_model/舊模型。

`$ python /SRC/retrain.py --image_dir /{your_new_training_set}/ --print_misclassified_test_images`

`$ mv /tmp/output_graph.pb /{your_TF_run/hw_model/}/output_graph_HW.pb`


## 客製化docker容器
*可使用原始Dockerfile檔案，image版本號與repo帳號依照需要修改，例如：tf_docker_v3 --> tf_hw_mod_v1, lswdokcer --> {$your_docker_repo}*


*修改Dockerfile

    $ vim Dockerfile
    
  依照需求修改設定，以ubuntu1604為系統基礎，由於功能已由pyinstaller封裝，因此docker image中不需再安裝任何套件。
  
    # TF run on Docker NCHC 2019-10-05 
    FROM ubuntu:16.04
    RUN apt-get -y update && apt-get install -y python3-tk
    COPY TF_run /home/TF_run
    WORKDIR /home/TF_run
    CMD ./Check.exe /home/TF_io/
    RUN echo "tf_hw_mod_v1 2019-10-05"

* 由Dockerfile建立container image

    `$ sudo docker build -t tf_hw_mod_v1 . `

* 設定該image版本號

    `$ sudo docker tag tf_hw_mod_v1 lswdokcer/tf_hw_mod_v1:latest`
    
* 登入您個人的docker帳號

    `$ sudo docker login`

* 將新的docker image上傳至repository

    `$ sudo docker push lswdokcer/tf_hw_mod_v1`

* 運行新的docker服務
    
    `$ sudo docker run -v /$HOST/TF_io:/home/TF_io lswdokcer/tf_hw_mod_v1`
    
***
# TF_io/out/ 輸出資料說明

![Alt text](https://github.com/vscv/HuWeiSPClassification/blob/master/SRC/DL_%E6%99%BA%E6%85%A7%E5%9C%92%E5%8D%80%E8%B7%AF%E9%9D%A2%E7%A9%8D%E6%B7%B9%E6%B0%B4%E5%81%B5%E6%B8%AC_TF_io_out_%E8%BC%B8%E5%87%BA%E8%B3%87%E6%96%99%E8%AA%AA%E6%98%8E.jpg)

* inf:輸出各類別機率分數
* jpg:原始影像
* \_cntp_txt:輪廓點座標(Red)(可重新繪圖使用)
* \_roi_seg_.png:套疊原始影像、虛擬警戒點、水域輪廓

***
