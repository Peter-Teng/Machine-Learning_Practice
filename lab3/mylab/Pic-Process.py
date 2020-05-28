from PIL import Image

for i in range(0,500):
    image = Image.open("lab3\ML2019-lab-03-master\datasets\original\\nonface\\nonface_%03d.jpg"%i)
    image = image.convert('L')
    image.thumbnail((24,24),resample=Image.BICUBIC)
    image.save('F:\机器学习\lab3\ML2019-lab-03-master\data\\nonface\\non_%03d.jpg'%i)