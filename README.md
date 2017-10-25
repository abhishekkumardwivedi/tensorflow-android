<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>

-----------------

| **`Android`** |

## Installation
Setup build system (ubuntu)
* Install bazel build system
```
sudo apt-get install openjdk-8-jdk
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update && sudo apt-get install oracle-java8-installer
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel
sudo apt-get upgrade bazel
```

* Get tensorflow source
```
cd ~
git clone https://github.com/abhishekkumardwivedi/tensorflow.git
```
In andorid studio
```
File -> open -> [~/tensorflow/tensorflow/examples/android]
```


## References

* https://github.com/tensorflow/tensorflow
* https://www.bazel.build/

## License

[Apache License 2.0](LICENSE)

