function [model] = SVR(ice2_train,vol2_train,CrossValidation,P )
%SVR �˴���ʾ�йش˺�����ժҪ
%  CrossValidation ������֤������P:����e -SVR����ʧ����P��ֵ
%  Ѱ�����c����/g����
[c,g] = meshgrid(-10:0.5:10,-10:0.5:10);
[m,n] = size(c);
cg = zeros(m,n);
eps = 10^(-4);%�����
bstc=0;
bstg=0;
minerror=Inf;
bstp=0;
tic

   for i = 1:m
    for j = 1:n
        %5�۽�����֤����˹�˺������ͷ�ϵ��2^c(i,j),�˺����е�gamma��������2^g(i,j)��eSVR
        %����e -SVR����ʧ����p��ֵ0.23
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
% ����/ѵ��SVM  
cmd = [' -t 2',' -c ',num2str(bstc),' -g ',num2str(bstg),' -s 3 -p 0.005'];
model = svmtrain(ice2_train,vol2_train,cmd);

end

