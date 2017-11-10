# darknet-people-count
It is a project to count people passed by the camera based on darknet people detection

# Requirement
linux

opencv

gpu, cuda7.5(necessary in consideration of speed)	

make 
and then prepare the weights file and video to be detected, which can be downloaded from here:

http://pan.baidu.com/s/1o8C8thK

# Run
./darknet detector demo cfg/coco.data cfg/yolo.cfg yolo.weights 1103_1.mov
