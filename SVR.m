function [model] = SVR(ice2_train,vol2_train,CrossValidation,P )
%SVR 此处显示有关此函数的摘要
%  CrossValidation 交叉验证折数，P:设置e -SVR中损失函数P的值
%  寻找最佳c参数/g参数
[c,g] = meshgrid(-10:0.5:10,-10:0.5:10);
[m,n] = size(c);
cg = zeros(m,n);
eps = 10^(-4);%误差限
bstc=0;
bstg=0;
minerror=Inf;
bstp=0;
tic

   for i = 1:m
    for j = 1:n
        %5折交叉验证，高斯核函数，惩罚系数2^c(i,j),核函数中的gamma函数设置2^g(i,j)，eSVR
        %设置e -SVR中损失函数p的值0.23
        cmd = ['-v ',num2str(CrossValidation),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 ',' -p ',num2str(P)];
        cg(i,j) = svmtrain(ice2_train,vol2_train,cmd);
        if cg(i,j) < minerror
            minerror = cg(i,j);
            bstc = 2^c(i,j);
            bstg = 2^g(i,j);
        end
        if abs(cg(i,j) - minerror) <= eps && bstc > 2^c(i,j)
            minerror = cg(i,j);
            bstc = 2^c(i,j);
            bstg = 2^g(i,j);
        end
    end
   end

toc
%%
% 创建/训练SVM  
cmd = [' -t 2',' -c ',num2str(bstc),' -g ',num2str(bstg),' -s 3 -p 0.005'];
model = svmtrain(ice2_train,vol2_train,cmd);

end

