%% ��ջ�������
clear all
clc
%% ��������
% ��ϱ�
G2=csvread('2.csv',0,0 );%ĩβ�����
%G3=csvread('3.csv',0,0 );%�ϴ����
G4=csvread('4.csv',0,0 );%ĩβ���������
G6=csvread('6.csv',0,0 );
G7=csvread('7.csv',0,0 );
G8=csvread('8.csv',0,0 );
G9=csvread('9.csv',0,0 );
G10=csvread('10.csv',0,0 );%��С
G11=csvread('11.csv',0,0 );%��С
G12=csvread('12.csv',0,0 );%��Сͬ11

%% �Ա������ݽ��д��������Բ�ֵ��Ϊarctan/tan
Maxnumb=5.5;
dir=0;%dir=0/1 tan/arctan

G2=arctan(G2,Maxnumb,dir);
G4=arctan(G4,Maxnumb,dir);
G6=arctan(G6,Maxnumb,dir);
G7=arctan(G7,Maxnumb,dir);
G8=arctan(G8,Maxnumb,dir);
G9=arctan(G9,Maxnumb,dir);
G10=arctan(G10,Maxnumb,dir);
G11=arctan(G11,Maxnumb,dir);
G12=arctan(G12,Maxnumb,dir);

figure(1);
plot(G2(:,3));
hold on
plot(G4(:,3));
hold on
plot(G6(:,3));
hold on
plot(G7(:,3));
hold on
plot(G8(:,3));
hold on
plot(G9(:,3));
hold on
plot(G10(:,3));
hold on
plot(G11(:,3));
hold on
plot(G12(:,3));
saveas(gcf, './build/1-original-ice-thickness', 'png')
%����ѵ�����Ͳ��Լ�
train=[G6;G2;G8;G9;G4;G11;G12];
test=[G7;G10];

%��ȥ��ѹ��ʼֵ
train(:,1)=train(:,1)-0.21;
train(:,2)=train(:,2)-0.4892;   
test(:,1)=test(:,1)-0.21;
test(:,2)=test(:,2)-0.4892;


%% ��άPCA����SVRԤ�����
CrossValidation=5;
P=0.25;
[Predict_1test,error_1test] = model2( train,test,CrossValidation,P);

%% ��ͼ
figure(3)
plot(1:length(test(:,3)),test(:,3),'r-*',1:length(test(:,3)),Predict_1test,'b:o');
grid on
legend('��ʵֵ','Ԥ��ֵ')
xlabel('�������')
ylabel('����')
string_2 = {'��άPCASVRģ�Ͳ��Լ�Ԥ�����Ա�';
           ['mse = ' num2str(error_1test(2)) ' R^2 = ' num2str(error_1test(3))]};
title(string_2)
saveas(gcf, './build/3-2d-PCASVR', 'png')
%% ��ά���ɷַ���
testpridict=test;
testpridict(:,3)=Predict_1test;
%��ѵ�����ľ�ֵ�ͱ�׼��
[row_train,colum_train] = size(train);
[row_test,colum_test] = size(testpridict);

train_Vmean=mean(train(:,1:3));
train_Vstd=std(train(:,1:3));
p=zscore(train(:,1:3));
[COEFF,SCORE]=pca(p);%PCA
%�Բ��Լ���ȥѵ������ֵ������ѵ������׼��
z=(testpridict(:,1:3)-train_Vmean)./train_Vstd;
%�õ�ӳ���Ĳ��Լ�
Convert=z*COEFF;  
%cumsum(latent)./sum(latent);%�����ۼƹ�����
%3άPCA��ѵ�����Ͳ��Լ��ĵ�ѹֵ���£�������ֵ���䣡��
train(1:row_train,1:2)=SCORE(1:row_train,1:2);  
testpridict(1:row_test,1:2)=Convert(1:row_test,1:2); 

%% �ֱ�ȡ��ѵ�����Ͳ��Լ��ĵ�ѹ�ͱ���ֵ
vol_train=train(:,1:2);
ice_train=train(:,3);
vol_test=testpridict(:,1:2);
ice_test=testpridict(:,3);


%% ��ͼ
[volt,inputp] = mapminmax([vol_train;vol_test]');
volt=volt';

[icethickness,outputp] = mapminmax([ice_train;ice_test]');
icethickness = icethickness';
figure(2)
plot3(volt(:,1),volt(:,2),icethickness,'r-*')%icethickness,'b:o')
xlabel('v1')
ylabel('v2')
zlabel('����')
grid on;
saveas(gcf, './build/2-3d-ice-thickness', 'png')
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
model=SVR(ice_train,vol_train,CrossValidation,P);%ѵ��3άģ��
[Predict_2test,error_2test,prob_estimates2] = svmpredict(ice_test,vol_test,model);%���Լ���Ԥ��
Predict_2test= mapminmax('reverse',Predict_2test',outputps);%ע�����ת��
Predict_2test=Predict_2test';
%% ѭ�������㷨


for i=1:1

testrepeat=test;
testrepeat(:,3)=Predict_2test;
zrepeat=(testrepeat(:,1:3)-train_Vmean)./train_Vstd;
Convertrepeat=zrepeat*COEFF; 
testrepeat(1:row_test,1:2)=Convertrepeat(1:row_test,1:2); 
vol_testrepeat=testrepeat(:,1:2);
ice_testrepeat=testrepeat(:,3);

vol_testrepeat = mapminmax('apply',vol_testrepeat',inputps);
vol_testrepeat = vol_testrepeat';
ice_testrepeat = mapminmax('apply',ice_testrepeat',outputps);
ice_testrepeat = ice_testrepeat';
[Predict_2test,error_2test,prob_estimates2] = svmpredict(ice_testrepeat,vol_testrepeat,model);%���Լ���Ԥ��
Predict_2test= mapminmax('reverse',Predict_2test',outputps);%ע�����ת��
Predict_2test=Predict_2test';
end

%% ��ͼ

figure(4)
plot(1:length(Predict_1test(:,1)),Predict_1test(:,1),'r-*',1:length(Predict_1test(:,1)),Predict_2test,'b:o');
grid on
legend('��άԤ��ֵ','��άԤ��ֵ')
xlabel('�������')
ylabel('����')
string_2 = {'��άPCASVRģ�Ͳ��Լ�Ԥ����'};
title(string_2)
saveas(gcf, './build/4-3d-PCASVR-predict', 'png')
figure(5)
plot(1:length(test(:,3)),test(:,3),'r-*',1:length(test(:,3)),Predict_2test,'b:o');
grid on
legend('��ʵֵ','��άԤ��ֵ')
xlabel('�������')
ylabel('����')
string_2 = {'��άPCASVRģ�Ͳ��Լ�Ԥ����'};
title(string_2)
saveas(gcf, './build/5-3d-PCASVR-origin', 'png')