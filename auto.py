import sys
import os

# python salientrun.py img1.jpg 
# python salientrun.py img2.jpg
# python siameserun img1.jpg img2.jpg
# python middle.py ./result/0001_Salient.png 0001.jpg
image1 = sys.argv[1]
image2 = sys.argv[2]
lst = sys.argv[1].split('/')
image_name1 = lst[len(lst)-1].split('.')[0]
lst = sys.argv[2].split('/')
image_name2 = lst[len(lst)-1].split('.')[0]

cmd = 'python salientrun.py '+image1
os.system(cmd)
cmd = 'python salientrun.py '+image2
os.system(cmd)

cmd = 'python middle.py ./result/'+image_name1+'_Salient.png '+image1
os.system(cmd)
cmd = 'python middle.py ./result/'+image_name2+'_Salient.png '+image2
os.system(cmd)

cmd = 'python siameserun.py '+sys.argv[1]+' '+sys.argv[2]
os.system(cmd)

print ("This is the path of the image: ", sys.argv[1])
print ("This is the path of the image: ", sys.argv[2])
