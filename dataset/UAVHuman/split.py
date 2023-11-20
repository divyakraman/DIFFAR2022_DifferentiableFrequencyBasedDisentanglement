a = open('images_list.txt', 'r')
f_test = open('test2.txt', 'w')
f_train = open('train2.txt', 'w')
b = a.read().split('\n')
for c in b:
	gt_label = c[9:12]
	print(gt_label)
	if(gt_label==''):
		continue
	gt_label = int(gt_label[0]) * 100 + int(gt_label[1]) * 10 + int(gt_label[2]) * 1
	#CSV2
	if(gt_label==1 or gt_label==2 or gt_label==7 or gt_label==9 or gt_label==13 or gt_label==15 or gt_label==17 or gt_label==23 or gt_label==25 or gt_label==27 or gt_label==33 or gt_label==34 or gt_label==41 or gt_label==42 or gt_label==48 or gt_label==50 or gt_label==51 or gt_label==53 or gt_label==55 or gt_label==58 or gt_label==65 or gt_label==68 or gt_label==76 or gt_label==82 or gt_label==85 or gt_label==90 or gt_label==105):
	#CSV1
	#if(gt_label==1 or gt_label==3 or gt_label==4 or gt_label==9 or gt_label==22 or gt_label==23 or gt_label==24 or gt_label==31 or gt_label==41 or gt_label==54 or gt_label==58 or gt_label==60 or gt_label==66 or gt_label==72 or gt_label==74 or gt_label==75 or gt_label==91 or gt_label==92 or gt_label==93 or gt_label==94 or gt_label==95 or gt_label==96 or gt_label==97 or gt_label==99 or gt_label==101 or gt_label==104 or gt_label==107 or gt_label==108 or gt_label==109 or gt_label==113):
		f_test.write(c+'\n')
	else:
		f_train.write(c+'\n')
a.close()
f_train.close()
f_test.close()