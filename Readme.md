Project summary:
This project was sent by Volley as an assignemnt for their Senior Computer Vision Engineer Role. 
It involves tracking a tennis ball across a board split into four regions. 
The main file in this repo reads in a video, finds a ball, calculates its center, determines what board region it is in, and saves an output video. 

Usage:
The solution presented will only work on other videos if those video are taken under the same lighting condtions with a similar colored ball. 

Dependencies / Prerequisites:
The dependecies for this project are the opencv libraries version 4.5.2. They can be found at https://opencv.org/
I used opencv 4.5.2 but 

Installation and execution instructions:
1. Set an enviroment variable called OPENCV_DIR to your OpenCV build directory i.e. "C:\OpenCV\opencv4.5.2\build". 
2. Pull project from git
3. Open a command prompt and navigate to the folder
4. Use cmake to build project and .exe. I used:
	cmake CmakeLists.txt -G "Visual Studio 15 2017 Win64"
	cmake --build .
5. Navigate to the bin folder: 
	cd bin
6. Run exe:
	Volley.exe yourVideo.mov
   where yourVideo.mov is the video you want to process
The video saved will be called "Result.mp4" and bve saved in the bin folder
	
	