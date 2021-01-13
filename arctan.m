function [ Data ] = arctan(Data,Maxnumb,dir)
%ACTAN 此处显示有关此函数的摘要
% Data 为数据，前面为电压值，此函数不会改变。
%最后一列为冰厚值，用于插值 Maxnumb用于归一化，务必大于最厚厚度,且冰厚单调上升，默认从0开始
[row,colum]=size(Data);
max=Data(row,colum);
maxp=(max*pi)/(2*Maxnumb);%转化为0-pi/2之间的数
if(dir==1)

 step=max*Maxnumb/(row-1);%求步长
 k=max/atan((row-1)*step);
 for i=1:row
    Data(i,3)=atan((i-1)*step);
 end
 %反映射
 Data(:,3)=k*Data(:,3);
end 
if(dir==0)

   step=maxp/(row-1);
   k=max/tan((row-1)*step);
   for i=1:row
   Data(i,3)=tan((i-1)*step);
   end
   %反映射
   Data(:,3)=k*Data(:,3);
 
 end      
