function [ Predict_1test,error_1test] = model2(train,test,CrossValidation,P)
%MODEL2 此处显示有关此函数的摘要
%   此处显示详细说明
%% 主成分分析
%求训练集的均值和标准差
%对两路电压V1 V2
[row_train,colum_train] = size(train);
[row_test,colum_test] = size(test);

train_Vmean=mean(train(:,1:2));
train_Vstd=std(train(:,1:2));
p=zscore(train(:,1:2));%对每一列标准化，即对V1 V2分别进行标准化
[COEFF,SCORE]=pca(p);%PCA
%对测试集减去训练集的电压均值并除以训练集标准差
z=(test(:,1:2)-train_Vmean)./train_Vstd;
%转换矩阵用法 注意z是去了均值的
%得到映射后的测试集
Convert=z*COEFF;  
%cumsum(latent)./sum(latent);%特征累计贡献率
%先2维PCA将训练集和测试集的电压值更新
train(1:row_train,1:2)=SCORE(1:row_train,1:2);   %取前2个主成分
test(1:row_test,1:2)=Convert(1:row_test,1:2); 

%% 分别取出训练集和测试集的电压和冰厚值
vol_train=train(:,1:2);
ice_train=train(:,3);
vol_test=test(:,1:2);
ice_test=test(:,3);

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
model=SVR(ice_train,vol_train,CrossValidation,P);%先训练2维模型

%利用模型1 SVM仿真预测
%[Predict_1train,error_1train] = svmpredict(ice1_train,vol1_train,model1);%训练集的一个预测
[Predict_1test,error_1test,prob_estimates1] = svmpredict(ice_test,vol_test,model);%测试集的预测
%反归一化
%predict_1trian = mapminmax('reverse',Predict_1train',outputps);
Predict_1test= mapminmax('reverse',Predict_1test',outputps);
Predict_1test=Predict_1test';
end

