import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.fft as fft

def numpy_sigmoid(x):
	return (1/(1+np.exp(x)))

def mask_pred(x):
	shape = x.shape	
	frequencies = fft.fftfreq(shape[2]).cuda()
	fft_compute = fft.fft(x, dim=2, norm='ortho').abs()
	frequencies = frequencies.unsqueeze(1)
	frequencies = frequencies.unsqueeze(1)
	frequencies = frequencies.unsqueeze(0)
	frequencies = frequencies.unsqueeze(0)	
	frequencies = frequencies * frequencies
	fft_compute = fft_compute * fft_compute
	mask1 = frequencies * fft_compute 
	mask2 = (1/(1+frequencies)) * fft_compute	
	mask = mask1 + mask2 
	
	mask = torch.sum(mask, 1)
	mask = torch.unsqueeze(mask, 1)
	mask_shape = mask.shape
	mask = mask.view(mask_shape[0], 1, mask_shape[2],mask_shape[3]*mask_shape[4])
	mask = F.softmax(mask, dim=3)
	mask = mask.view(mask_shape[0], 1, mask_shape[2],mask_shape[3],mask_shape[4])
	
	return mask

def frameSampler1(all_frames, uniform_frames, num_frames, model_link):
	frame_shape = all_frames.shape
	new_frames = torch.zeros(frame_shape[0], num_frames, 3, frame_shape[3], frame_shape[4])
	step = int(frame_shape[1]/num_frames)
	if(step<=0):
		return uniform_frames
	else: 		
		model = torch.load(model_link, map_location = 'cpu').cuda()
		all_frames = all_frames.permute(0,2,1,3,4).cuda()
		uniform_frames = uniform_frames.permute(0,2,1,3,4).cuda() #Batch size x 3 x frames X H x W
		#Uniform frames output
		uniform_frames_output = model.module.model_backbone.conv1(uniform_frames)
		uniform_frames_output = model.module.model_backbone.bn1(uniform_frames_output)
		uniform_frames_output = model.module.model_backbone.relu(uniform_frames_output)
		uniform_frames_output = model.module.model_backbone.maxpool(uniform_frames_output)
		uniform_frames_output = model.module.model_backbone.layer1(uniform_frames_output)
				
		'''
		uniform_frames_output = model.module.conv1_s(uniform_frames)
		uniform_frames_output = model.module.conv1_t(uniform_frames_output)
		uniform_frames_output = model.module.bn1(uniform_frames_output)
		uniform_frames_output = model.module.relu(uniform_frames_output)
		uniform_frames_output = model.module.layer1(uniform_frames_output)
		'''
		#uniform_frames_output = model.module.layer2(uniform_frames_output)
		#uniform_frames_output = model.module.layer3(uniform_frames_output)
		
		upsample_shape = uniform_frames_output.shape
		#Uniform frames frequency
		mask = mask_pred(uniform_frames_output).detach()
		uniform_frames_output = uniform_frames_output * mask
		uniform_frames_output = uniform_frames_output.detach()
		uniform_frames_output = torch.sum(uniform_frames_output, 1) #Batchsize, T, H, W
		#uniform_frames_output = fft.fft2(uniform_frames_output, dim=(2,3), norm='ortho').abs()
		for i in range(0,num_frames):
			scores = []
			running_count = step * i
			chosen_frame = running_count 
			for j in range(running_count, running_count+step):
				current_frame_output = model.module.model_backbone.conv1(all_frames[:,:,j,:,:].unsqueeze(2))
				current_frame_output = model.module.model_backbone.bn1(current_frame_output)
				current_frame_output = model.module.model_backbone.relu(current_frame_output)
				current_frame_output = model.module.model_backbone.maxpool(current_frame_output)
				current_frame_output = model.module.model_backbone.layer1(current_frame_output)
				
				'''
				current_frame_output = model.module.conv1_s(all_frames[:,:,j,:,:].unsqueeze(2))
				current_frame_output = model.module.conv1_t(current_frame_output)
				current_frame_output = model.module.bn1(current_frame_output)
				current_frame_output = model.module.relu(current_frame_output)
				current_frame_output = model.module.layer1(current_frame_output)
				'''
				#current_frame_output = model.module.layer2(current_frame_output)
				#current_frame_output = model.module.layer3(current_frame_output)
				
				current_frame_output = current_frame_output * mask[:,:,i,:,:].unsqueeze(2)
				current_frame_output = current_frame_output.detach()
				current_frame_output = torch.sum(current_frame_output, 1) 
				prior_frame_importance = 0
				for k in range(0,i):
					correlation = torch.mean((current_frame_output[:,0,:,:] - uniform_frames_output[:,k,:,:])**2).abs()
					prior_frame_importance = prior_frame_importance + ((num_frames - i+k)/num_frames) * correlation.cpu().numpy()
			
				spot_price = prior_frame_importance 
				#Strike price
				strike_price = 0
				for k in range(i+1, num_frames):
					correlation = torch.mean((current_frame_output[:,0,:,:] - uniform_frames_output[:,k,:,:])**2).abs()
					#strike_price = strike_price + ((num_frames-k+1)/num_frames) * correlation
					strike_price = strike_price + ((num_frames - k+ i)/num_frames) * correlation.cpu().numpy()
				#Applying the formula
				t = num_frames - i
				d1 = (spot_price/(spot_price+strike_price))*i
				d2 = (strike_price/(spot_price+strike_price))*t
				cost = d1*spot_price + d2*strike_price
				#cost = spot_price + strike_price

				scores.append(cost)
				if(j==running_count):
					chosen_frame_cost = cost
				if(cost>chosen_frame_cost):
					chosen_frame = j
					chosen_frame_cost = cost
			#Assign new frame
			for video_num in range(0,frame_shape[0]):
				new_frames[video_num,i,:,:,:] = all_frames[video_num,:,chosen_frame,:,:].cpu()
			#print(chosen_frame)
		return new_frames