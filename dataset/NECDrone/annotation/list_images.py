import pandas as pd 

list_images = pd.read_csv('NEC-Drone-16_testlist_release.csv')
f = open('test16.txt', 'w')

num_images = len(list_images)-1
a = list_images.values

#Folder path, action label

for b in range(0,num_images):
	c = a[b,0]+'/'+a[b,1]+'/'+a[b,2]+'/'+a[b,3]+'/'
	#c = c + '\t' + str(a[b,4]) + '\t' + str(a[b,5]) + '\t' + str(a[b,6]) + '\n'
	c = c + '\t' + str(a[b,6]) + '\n'
	f.write(c)

f.close()

print(a)
