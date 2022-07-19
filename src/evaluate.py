# -*- coding: utf-8 -*-
# @Date    : 2017-12-26 20:28:33
# @Author  : Liang, Peifeng (liangpeifeng@akane.waseda.jp)
# @Link    : ${link}
# @Version : $Id$
'''
Return value matrix:
TP     FP  FN  TN      TD     FD
Recall Pre TNR F-score G-mean Acc
'''
import os
import numpy as np
class evaluation:
	"""docstring for ev"""
	def __init__(self, p, lable):
		self.p = p
		self.l =lable

	def evalue(self):
		if self.p.shape[0]!=self.l.shape[0]:
			raise ValueError("size two arraies are not equate!!" )
		num = self.p.shape[0]
		TP=0.0
		TN=0.0
		TD=0.0
		FP=0.0
		FN=0.0
		FD=0.0
		for i in range(num):
			if self.p[i]==self.l[i]:	
				TD +=1
				if self.l[i]==1:
					TP +=1
				else:
					TN +=1
			else:
				FD +=1
				if self.l[i]==1:
					FN +=1
				else:
					FP +=1
		print('TP: %.1f FP: %.1f FN: %.1f TN: %.1f TD: %.1f FD: %.1f ' % (TP, FP, FN, TN, TD, FD))
		Acc = TD/num
		Recall = TP/(TP+FN)
		if TP+FP==0:
			Prec=np.NaN
			F_score=0
		else:
			Prec =TP / (TP+FP)
			F_score = 2*Recall*Prec/(Recall+Prec)
		TNR = TN/(TN+FP)
		
		G_mean = np.square(Recall*TNR)
		if TP+FP==0:
			print('Recall: %.4f Pre: NaN TNR: %.4f F-score: %.4f G-mean: %.4f Acc: %.4f' % (Recall, TNR, F_score, G_mean, Acc))
		else:
			print('Recall: %.4f Pre: %.4f TNR: %.4f F-score: %.4f G-mean: %.4f Acc: %.4f' % (Recall, Prec, TNR, F_score, G_mean, Acc))
		value = np.zeros((2,6))
		value[0,0] = TP
		value[0,1] = FP
		value[0,2] = FN
		value[0,3] = TN
		value[0,4] = TD
		value[0,5] = FD
		
		value[1,0] = Recall
		value[1,1] = Prec
		value[1,2] = TNR
		value[1,3] = F_score
		value[1,4] = G_mean
		value[1,5] = Acc
		return value
		'''
	def evaluation(predict_lables, data_lables):
		import numpy as np
		if predict_lables.shape[0]!=data_lables.shape[0]:
			raise ValueError("size two arraies are not equate!!" )
		num = predict_lables.shape[0]
		TP=0.0
		TN=0.0
		TD=0.0
		FP=0.0
		FN=0.0
		TD=0.0
		for i in xrange(num):
			if predict_lables[i]==data_lables[i]:	
				TD +=1
				if data_lables[i]==1:
					TP +=1
				else:
					TN +=1
			else:
				FD +=1
				if data_lables[i]==1:
					FN +=1
				else:
					FP +=1

		Acc = TD/num
		Recall = TP/(TP+FN)
		Pre =TP / (TP+FP)
		TNR = TN/(TN+FP)
		F_score = 2*Recall*Pre/(Recall+Pre)
		G_mean = np.square(Recall*TNR)
		print('TP: %d FP: %d FN: %d TN: %d TD: %d FD: %d ' % (TP, FP, FN, TN, TD, FD))
		print('Recall: %.2f Pre: %.2f TNR: %.2f F-score: %.2f G-mean: %.2f Acc: %.2f' % (Recall, Pre, TNR, F_score, G_mean, Acc))
		value = np.zeros((2,6))
		value[0,0] = TP
		value[0,1] = FP
		value[0,2] = FN
		value[0,3] = TN
		value[0,4] = TD
		value[0,5] = FD
		
		value[1,0] = Recall
		value[1,1] = Pre
		value[1,2] = TNR
		value[1,3] = F_score
		value[1,4] = G_mean
		value[1,5] = Acc
	'''