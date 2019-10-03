# HuWeiSPClassification
Check and send new images to tf inception v3 model, then output the label.


# 使用方法
## 執行辨識docker
sudo docker run -v /$HOST/TF_io:/home/TF_io lswdokcer/tf_docker_v3

* tf_docker_v3: v[] 為版本號
* /$HOST替：請將/$HOST替換成host上的資料夾
* lswdokcer: 為docker hub存放空間名稱

# 客製化需求
## 安裝Docker環境
$ sudo apt-get update 
$ sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo apt-key fingerprint 0EBFCD88
$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs)  stable"

INSTALL DOCKER
$ sudo apt-get update
$ sudo apt-get install docker-ce (for Docker CE)
Check version of docker by: $ apt-cache madison docker-ce

## 客製化docker容器

## 功能增加與修改
