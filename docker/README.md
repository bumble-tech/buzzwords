# Docker Installation

To deploy Buzzwords in some use cases, we use Docker images with the library already installed in a conda environment. If you're struggling to install buzzwords, this is an alternative option to run your training scripts

## Building a Base Image

To build a base image, run this from the base directory of the repo

```sh
docker build \
	-t buzzwords-base:$VERSION \
	-f docker/Dockerfile \
	.
```

This will build the docker image (takes 30+ minutes to install, just let it run)

<div align='center'>
	<table>
		<th>
		:warning: Buzzwords is installed in the `buzzwords` conda environment in these images
		</th>
	</table>
</div>

***
## Test Deployment

There is also a simple script to run a test of the base image. This will build a simple image that checks if Buzzwords has actually been installed. The syntax is the same as for building the base image

```sh
$ chmod +x docker/test/run_test.sh
$ docker/test/run_test.sh $IMAGE
```

If the base image works properly, you will see something like:

```
[sofarrell@datascience6.mlan buzzwords]$ docker/test/run_test.sh buzzwords-base:0.2.2
Sending build context to Docker daemon  4.096kB
Step 1/5 : ARG IMAGE=placeholder
Step 2/5 : FROM $IMAGE
 ---> c874ff640477
Step 3/5 : COPY basic_test.sh .
 ---> Using cache
 ---> db3b1a59f6f5
Step 4/5 : RUN chmod +x basic_test.sh
 ---> Using cache
 ---> 34b916e4543b
Step 5/5 : CMD bash basic_test.sh
 ---> Running in be6f290c1a7c
Removing intermediate container be6f290c1a7c
 ---> 40a4f42f7196
Successfully built 40a4f42f7196
Successfully tagged test-buzzwords:latest
Success! Buzzwords was successfully installed to this image
```

You can then use this docker image to run your scripts. You simply mount your directory to the docker image with `-v`

```sh
sudo docker run \
    -v /home:/home \
    -w `pwd` \
    buzzwords \
    ./basic_test.sh
```

Just remember to either change the Dockerfile to start conda by default or wrap your python in a bash script that activate conda first
```sh
source activate buzzwords

python3 train.py
```