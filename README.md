# HuWeiSPClassification
Check and send new images to tf inception v3 model, then output the label.


# 使用方法
## 執行辨識docker
sudo docker run -v /$HOST/TF_io:/home/TF_io lswdokcer/tf_docker_v3

* tf_docker_v3: v[] 為版本號
* $HOST：請將/$HOST替換成您主機上的資料夾路徑
* TF_io: 固定名稱，存放新收影像檔與辨識結果
* lswdokcer: 為docker hub存放空間之名稱

# 客製化需求
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

## 辨識模組的Docker image使用
1. 若host主機已安裝好docker engine，另外還需建立輸出影像與輸出結果的資料夾。例如：host主機帳號下建立/host/name/$TF_io_root_dir/TF_io主資料夾，並在其下建立/in與/out資料夾。

2. 使用前請先確認每個鏡頭的ROI設定檔(00xxxxxlive.cfg)已放入host/name/$TF_io_root_dir/主資料夾中。當container 運行時，會自動將新加入TF_io/in/中的影像進行辨識，其中水域偵測需要依據每個鏡頭的ROI設定檔(00xxxxxlive.cfg)來進行處理，最後才輸出至TF_io/out/中。

$TF_io/主資料夾內容結構如下：

    in/
    out/
    001140live.cfg
    002051live.cfg
    009052live.cfg
    ...

## 客製化docker容器

## 功能增加與修改
