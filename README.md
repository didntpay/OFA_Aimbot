# OFA_Aimbot
Aim bot that doesn't require finding the game's memory address nor anything internal.

# Abstract
This project is for study purpose only, not for commercial uses. 
In order for the scripts in main folder to work, you must download python(3.6 is the version I am using) as well as modules listed below
  * Tensorflow(GPU recommended)
    * Additional packages are required. Please checkout https://www.tensorflow.org/install/gpu for more
  * win32gui
  * numpy
  * urllib
  * Opencv

# Theory
In the development of hacks, modifying the game's memory address is very tedious, espeically with all the anti-cheats. So from a object detection perspective, the hack might not be as powerful as directly aim locking but it is harder for the anti-cheat to detect. 

# Remark
This scripts require some level of hardware to run. I am currently having 10 frame per second with GTX-950M and I7-xxxx. The higher the better. The first time you run this script, please add in the code from RunFirstTime.py to download the object detection model from tesorflow.

# Result
With object_detection model(light) developed by Tensorflow, the hack is able to pick out enemy in your visable range and lock on them.
![](OFA.gif)
