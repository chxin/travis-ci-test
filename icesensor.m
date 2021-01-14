%% 清空环境变量
clear all
clc
%% 导入数据
% 混合冰
G2=csvread('2.csv',0,0 );%末尾有误差
%G3=csvread('3.csv',0,0 );%较大误差
G4=csvread('4.csv',0,0 );%末尾上翘有误差
G6=csvread('6.csv',0,0 );
G7=csvread('7.csv',0,0 );
G8=csvread('8.csv',0,0 );
G9=csvread('9.csv',0,0 );
G10=csvread('10.csv',0,0 );%较小
G11=csvread('11.csv',0,0 );%较小
G12=csvread('12.csv',0,0 );%较小同11

%% 对冰厚数据进行处理：将线性插值换为arctan/tan
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
%划分训练集和测试集
train=[G6;G2;G8;G9;G4;G11;G12];
test=[G7;G10];

%减去电压初始值
train(:,1)=train(:,1)-0.21;
train(:,2)=train(:,2)-0.4892;   
test(:,1)=test(:,1)-0.21;
test(:,2)=test(:,2)-0.4892;


%% 二维PCA处理及SVR预测冰厚
CrossValidation=5;
P=0.25;
[Predict_1test,error_1test] = model2( train,test,CrossValidation,P);

%% 作图
figure(3)
plot(1:length(test(:,3)),test(:,3),'r-*',1:length(test(:,3)),Predict_1test,'b:o');
grid on
legend('真实值','预测值')
xlabel('样本编号')
ylabel('冰厚')
string_2 = {'二维PCASVR模型测试集预测结果对比';
           ['mse = ' num2str(error_1test(2)) ' R^2 = ' num2str(error_1test(3))]};
title(string_2)
saveas(gcf, './build/3-2d-PCASVR', 'png')
%% 三维主成分分析
testpridict=test;
testpridict(:,3)=Predict_1test;
%求训练集的均值和标准差
[row_train,colum_train] = size(train);
[row_test,colum_test] = size(testpridict);

train_Vmean=mean(train(:,1:3));
train_Vstd=std(train(:,1:3));
p=zscore(train(:,1:3));
[COEFF,SCORE]=pca(p);%PCA
%对测试集减去训练集均值并除以训练集标准差
z=(testpridict(:,1:3)-train_Vmean)./train_Vstd;
%得到映射后的测试集
Convert=z*COEFF;  
%cumsum(latent)./sum(latent);%特征累计贡献率
%3维PCA将训练集和测试集的电压值更新，而冰厚值不变！！
train(1:row_train,1:2)=SCORE(1:row_train,1:2);  
testpridict(1:row_test,1:2)=Convert(1:row_test,1:2); 

%% 分别取出训练集和测试集的电压和冰厚值
vol_train=train(:,1:2);
ice_train=train(:,3);
vol_test=testpridict(:,1:2);
ice_test=testpridict(:,3);


%% 作图
[volt,inputp] = mapminmax([vol_train;vol_test]');
volt=volt';

[icethickness,outputp] = mapminmax([ice_train;ice_test]');
icethickness = icethickness';
figure(2)
plot3(volt(:,1),volt(:,2),icethickness,'r-*')%icethickness,'b:o')
xlabel('v1')
ylabel('v2')
zlabel('冰厚')
grid on;
saveas(gcf, './build/2-3d-ice-thickness', 'png')
%% 数据归一化
% 输入电压
[vol_train,inputps] = mapminmax(vol_train');%对电压值的两个特征都分别行归一化处理
vol_train = vol_train';
vol_test = mapminmax('apply',vol_test',inputps);
vol_test = vol_test';

% 输出冰厚
[ice_train,outputps] = mapminmax(ice_train');
ice_train = ice_train';
ice_test = mapminmax('apply',ice_test',outputps);
ice_test = ice_test';

%% SVR模型创建/训练
model=SVR(ice_train,vol_train,CrossValidation,P);%训练3维模型
[Predict_2test,error_2test,prob_estimates2] = svmpredict(ice_test,vol_test,model);%测试集的预测
Predict_2test= mapminmax('reverse',Predict_2test',outputps);%注意矩阵转置
Predict_2test=Predict_2test';
%% 循环迭代算法


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
[Predict_2test,error_2test,prob_estimates2] = svmpredict(ice_testrepeat,vol_testrepeat,model);%测试集的预测
Predict_2test= mapminmax('reverse',Predict_2test',outputps);%注意矩阵转置
Predict_2test=Predict_2test';
end

%% 作图

figure(4)
plot(1:length(Predict_1test(:,1)),Predict_1test(:,1),'r-*',1:length(Predict_1test(:,1)),Predict_2test,'b:o');
grid on
legend('二维预测值','三维预测值')
xlabel('样本编号')
ylabel('冰厚')
string_2 = {'三维PCASVR模型测试集预测结果'};
title(string_2)
saveas(gcf, './build/4-3d-PCASVR-predict', 'png')
figure(5)
plot(1:length(test(:,3)),test(:,3),'r-*',1:length(test(:,3)),Predict_2test,'b:o');
grid on
legend('真实值','三维预测值')
xlabel('样本编号')
ylabel('冰厚')
string_2 = {'三维PCASVR模型测试集预测结果'};
title(string_2)
saveas(gcf, './build/5-3d-PCASVR-origin', 'png')