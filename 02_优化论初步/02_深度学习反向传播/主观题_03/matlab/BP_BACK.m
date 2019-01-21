%-------------函数说明----------------  
%    交换地点顺序  
%       输入变量：  
%               training_example:待训练的数据  
%                    eta： 学习速率  
%---------------------------------------  
function BP_BACK(training_example,eta)  
[m,n] = size(training_example); %m--行，n--列  
%初始化权值矩阵-0.5~0.5之间  
w = rand(2,3) - 0.5;      
v = rand(3,2) - 0.5;  
u = rand(2,3) - 0.5;  
%------------------------  
for num = 1:n  %对每组输入量与输出量  
%%----------------------------------------  
        one_sample = training_example(:,num); %取一组  
        x = one_sample(1:3);        %提取其中的输入  
        y = one_sample(4:5);        %提取其中的输出  
        net2 = w * x;               %第一层求和值  
        for i=1:2  
             hidden1(i)=1/(1+exp(-net2(i)));%进行sigmoid处理输出  
        end  
        net3 = v * hidden1';      %第二层求和值   
        for i=1:3  
             hidden2(i)=1/(1+exp(-net3(i)));%进行sigmoid处理输出  
        end  
        net4 = u * hidden2';      %第三层求和值  
        for i=1:2  
            o(i)=1/(1+exp(-net4(i)));%sigmoid处理输出，最终的输出值o  
        end  
%%-------------反向传播算法，计算各层delta值（误差E对各层权值的导数）-----------------  
       %最后一层delta值  
        for i = 1:2  
             delta3(i) = (y(i)-o(i))*o(i)*(1-o(i)); %计算公式  
        end  
       %-----第二个隐含层---  
        for j = 1:3      %计算公式，与其后一层的delta值相关  
             delta2(j) = hidden2(j)*(1-hidden2(j))*delta3*u(:,j);  
        end  
       %-----第一个隐含层---  
        for k = 1:2      %计算公式，与其后一层的delta值相关  
             delta1(k) = hidden1(k)*(1-hidden1(k))*delta2*v(:,k);  
        end  
%---------------------------------------------------------  
%--------各层delta计算完后开始更新权值---------------------  
%---更新u权值-----  
        for i = 1:2   
               for j = 1:3      %计算公式  w = w + eta*delta*x  
                     u(i,j) = u(i,j) + eta*delta3(i)*hidden2(j);  
               end  
        end  
%---更新v权值-----  
        for i = 1:3  
                for j = 1:2  
                     v(i,j) = v(i,j) + eta*delta2(i)*hidden1(j);  
                end  
        end  
%---更新w权值-----  
        for i = 1:2  
                for j = 1:3  
                     w(i,j) = w(i,j) + eta*delta1(i)*x(j);  
                end  
        end  
%---------------------------------------------------------  
%--------------记录一下这个过程后的误差值  
            e=o'-y;%计算误差向量  （计算输出-目标输出）  
            sigma(num)=e'*e;%计算误差平方和  
end  
%%-------------------------------------------------------  
plot(sigma); 