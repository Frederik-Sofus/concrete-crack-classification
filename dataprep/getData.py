import os

os.system("wget  https://data.mendeley.com/datasets/5y9wdsg2zt/1/files/c0d86f9f-852e-4d00-bf45-9a0e24e3b932/Concrete%20Crack%20Images%20for%20Classification.rar?dl=1")

print("Downloaded data from url")

os.system("mv  Concrete\ Crack\ Images\ for\ Classification.rar\?dl\=1 ConcreteCrackImages.rar")

print("Renamed rar file")

print(" ")
print("Installing unrar")

os.system("sudo apt-get install unrar")

os.system("mkdir data")
print("Installed unrar")
print("Start unraring files")
os.system("unrar x -r ConcreteCrackImages.rar /home/$USER/deep/data")
print("Succesfully unpacked and moved data to data folder")
