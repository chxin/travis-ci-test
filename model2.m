function [ Predict_1test,error_1test] = model2(train,test,CrossValidation,P)
%MODEL2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%% ���ɷַ���
%��ѵ�����ľ�ֵ�ͱ�׼��
%����·��ѹV1 V2
[row_train,colum_train] = size(train);
[row_test,colum_test] = size(test);

train_Vmean=mean(train(:,1:2));
train_Vstd=std(train(:,1:2));
p=zscore(train(:,1:2));%��ÿһ�б�׼��������V1 V2�ֱ���б�׼��
[COEFF,SCORE]=pca(p);%PCA
%�Բ��Լ���ȥѵ�����ĵ�ѹ��ֵ������ѵ������׼��
z=(test(:,1:2)-train_Vmean)./train_Vstd;
%ת�������÷� ע��z��ȥ�˾�ֵ��
%�õ�ӳ���Ĳ��Լ�
Convert=z*COEFF;  
%cumsum(latent)./sum(latent);%�����ۼƹ�����
%��2άPCA��ѵ�����Ͳ��Լ��ĵ�ѹֵ����
train(1:row_train,1:2)=SCORE(1:row_train,1:2);   %ȡǰ2�����ɷ�
test(1:row_test,1:2)=Convert(1:row_test,1:2); 

%% �ֱ�ȡ��ѵ�����Ͳ��Լ��ĵ�ѹ�ͱ���ֵ
vol_train=train(:,1:2);
ice_train=train(:,3);
vol_test=test(:,1:2);
ice_test=test(:,3);

%% ���ݹ�һ��

% �����ѹ
[vol_train,inputps] = mapminmax(vol_train');%�Ե�ѹֵ�������������ֱ��й�һ������
vol_train = vol_train';
vol_test = mapminmax('apply',vol_test',inputps);
vol_test = vol_test';

% �������
[ice_train,outputps] = mapminmax(ice_train');
ice_train = ice_train';
ice_test = mapminmax('apply',ice_test',outputps);
ice_test = ice_test';

%% SVRģ�ʹ���/ѵ��
model=SVR(ice_train,vol_train,CrossValidation,P);%��ѵ��2άģ��

%����ģ��1 SVM����Ԥ��
%[Predict_1train,error_1train] = svmpredict(ice1_train,vol1_train,model1);%ѵ������һ��Ԥ��
[Predict_1test,error_1test,prob_estimates1] = svmpredict(ice_test,vol_test,model);%���Լ���Ԥ��
%����һ��
%predict_1trian = mapminmax('reverse',Predict_1train',outputps);
Predict_1test= mapminmax('reverse',Predict_1test',outputps);
Predict_1test=Predict_1test';
end

