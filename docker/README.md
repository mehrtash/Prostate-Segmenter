# README

## CPU

##### Building the docker
```
docker build -t prostate-segmenter-cpu -f Dockerfile.cpu .
```

##### Running the docker
```
docker run -t -v [Absolute PATH to the Project Folder]/Prostate-Segmenter/prostatesegmenter/data/test/:/home/deepinfer/data deepinfer/prostate-segmenter-cpu -i /home/deepinfer/data/input.nrrd -o /home/deepinfer/data/label_predicted_test.nrrd
```
