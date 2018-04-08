from skimage import io
import os

TRAINING_PATH = "F:\GitHub - Projects\Object-Detection-and-Recognition" \
                "\Data set\Training"
TESTING_PATH = "F:\GitHub - Projects\Object-Detection-and-Recognition" \
               "\Data set\Testing"

os.chdir(TRAINING_PATH)

Training_Pics = [
    'Model1 - Cat.jpg',
    'Model2 - Cat.jpg',
    'Model3 - Cat.jpg',
    'Model4 - Cat.jpg',
    'Model5 - Cat.jpg',
    'Model6 - Laptop.jpg',
    'Model7 - Laptop.jpg',
    'Model8 - Laptop.jpg',
    'Model9 - Laptop.jpg',
    'Model10 - Laptop.jpg',
    'Model11 - Apple.jpg',
    'Model12 - Apple.jpg',
    'Model13 - Apple.jpg',
    'Model14 - Apple.jpg',
    'Model15 - Apple.jpg',
    'Model16 - Car.jpg',
    'Model17 - Car.jpg',
    'Model18 - Car.jpg',
    'Model19 - Car.jpg',
    'Model20 - Car.jpg',
    'Model21 - Helicopter.jpg',
    'Model22 - Helicopter.jpg',
    'Model23 - Helicopter.jpg',
    'Model24 - Helicopter.jpg',
    'Model25 - Helicopter.jpg'
]
Training = {
    'Cat': [],
    'Laptop': [],
    'Apple': [],
    'Car': [],
    'Helicopter': []
}
# Reading Data From Training File
for i in range(25):
    img = io.imread(Training_Pics[i], as_grey=True)
    if i < 5:
        Training['Cat'].append(img)
    elif i < 10:
        Training['Laptop'].append(img)
    elif i < 15:
        Training['Apple'].append(img)
    elif i < 20:
        Training['Car'].append(img)
    elif i < 25:
        Training['Helicopter'].append(img)

print(Training)
