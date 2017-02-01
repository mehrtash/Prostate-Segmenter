# README

## CPU

##### Downloading the docker image from the hub
```
docker pull deepinfer/prostate-segmenter-cpu
```

##### Building the docker
```
docker build -t prostate-segmenter-cpu -f Dockerfile.cpu .
```

##### Running the docker
```
docker run -t -v [Absolute PATH to the Project Folder]/Prostate-Segmenter/prostatesegmenter/data/test/:/home/deepinfer/data deepinfer/prostate-segmenter-cpu -InputVolume /home/deepinfer/data/input.nrrd -OutputLabel /home/deepinfer/data/label_predicted_test.nrrd
```
