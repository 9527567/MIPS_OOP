# MIPS_OOP

requirement: Eigen 3.4, Boost, OpenMP
~~~bash
sudo apt install libboost-container-dev libeigen3-dev
mkdir  build
cd build
cmake ..
make
~~~

main 
-N 17770 -Q 999 -D 300 -K 100 -B 300 -S 10000 --fx "_X_17770_300.txt" --fq "_Q_999_300.txt" --up 1024 --s0 5 
