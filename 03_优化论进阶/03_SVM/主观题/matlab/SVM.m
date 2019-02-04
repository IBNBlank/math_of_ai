%Please first install cvx toolbox from http://cvxr.com



% load data
load q1x.dat
load q1y.dat
% define variables
X = q1x;
y = 2*(q1y-0.5);
C = 1;
m = size(q1x,1);
n = size(q1x,2);
% train svm using cvx
cvx_begin
    variables w(n) b xi(m)
    minimize 1/2*sum(w.*w) + C*sum(xi)
    y.*(X*w + b) >= 1 - xi;
    xi >= 0;
cvx_end

% visualize
xp = linspace(min(X(:,1)), max(X(:,1)), 100);
yp = - (w(1)*xp + b)/w(2);
yp1 = - (w(1)*xp + b - 1)/w(2); % margin boundary for support vectors for y=1
yp0 = - (w(1)*xp + b + 1)/w(2); % margin boundary for support vectors for y=0
idx0 = find(q1y==0);
idx1 = find(q1y==1);
plot(q1x(idx0, 1), q1x(idx0, 2), 'rx'); hold on

plot(q1x(idx1, 1), q1x(idx1, 2), 'go');
plot(xp, yp, '-b', xp, yp1, '--g', xp, yp0, '--r');
hold off
title(sprintf('decision boundary for a linear SVM classifier with C=%g', C));