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

figure(5)
plot(1:length(test(:,3)),test(:,3),'r-*',1:length(test(:,3)),Predict_2test,'b:o');
grid on
legend('��ʵֵ','��άԤ��ֵ')
xlabel('�������')
ylabel('����')
string_2 = {'��άPCASVRģ�Ͳ��Լ�Ԥ����'};
title(string_2)

