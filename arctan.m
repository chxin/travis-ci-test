function [ Data ] = arctan(Data,Maxnumb,dir)
%ACTAN �˴���ʾ�йش˺�����ժҪ
% Data Ϊ���ݣ�ǰ��Ϊ��ѹֵ���˺�������ı䡣
%���һ��Ϊ����ֵ�����ڲ�ֵ Maxnumb���ڹ�һ������ش��������,�ұ��񵥵�������Ĭ�ϴ�0��ʼ
[row,colum]=size(Data);
max=Data(row,colum);
maxp=(max*pi)/(2*Maxnumb);%ת��Ϊ0-pi/2֮�����
if(dir==1)

 step=max*Maxnumb/(row-1);%�󲽳�
 k=max/atan((row-1)*step);
 for i=1:row
    Data(i,3)=atan((i-1)*step);
 end
 %��ӳ��
 Data(:,3)=k*Data(:,3);
end 
if(dir==0)

   step=maxp/(row-1);
   k=max/tan((row-1)*step);
   for i=1:row
   Data(i,3)=tan((i-1)*step);
   end
   %��ӳ��
   Data(:,3)=k*Data(:,3);
 
 end      
