function [X,Y,Z] = sample_spherePoint()
close all

n = 50;
[theta,phi,X,Y,Z] = sample5(n,1);
[theta2,phi2,X2,Y2,Z2] = sample5(n,-1);

figure;
axis equal;
hold on;
plot(theta,phi,'.');
plot(theta2,phi2,'.');
xlabel('theta')
ylabel('phi')

figure;
axis equal;
hold on;
scatter3(X,Y,Z,'.');
scatter3(X2,Y2,Z2,'.');
xlabel('x-axis')
ylabel('y-axis')
zlabel('z-axis')
end

%% first try
function [theta,phi,X,Y,Z] = sample1(n)
theta = linspace(0,2*pi,2*n);
phi = linspace(0,pi,n);
len_theta = size(theta,2);
len_phi = size(phi,2);
theta = repmat(theta,1,len_phi);
phi = repmat(phi,len_theta,1);
phi = phi(:);
phi = phi';
X = sin(phi) .* cos(theta);
Y = sin(phi) .* sin(theta);
Z = cos(phi);
figure;
axis equal;
hold on;
scatter3(X,Y,Z,'.');
end

%% second try
function [theta,u,X,Y,Z] = sample2(n)
theta = linspace(0,2*pi,n); 
u = linspace(-1,1,n);
theta = repmat(theta,1,n);
u = repmat(u,n,1);
u = u(:);
u = u';
X = cos(theta).*sqrt(1-u.^2);
Y = sin(theta).*sqrt(1-u.^2);
Z = u;
end

%% third try
function [x1,x2,X,Y,Z] = sample3(n)
x1 = linspace(-1,1,n);
x2 = linspace(-1,1,n);
x1 = repmat(x1,1,n);
x2 = repmat(x2,n,1);
x2 = x2(:);
x2 = x2';
temp = x1.^2 + x2.^2;
temp(temp>=1) = 0;
temp = logical(temp);
x1 = x1(temp);
x2 = x2(temp);
X = 2*x1.*sqrt(1-x1.^2-x2.^2);
Y = 2*x2.*sqrt(1-x1.^2-x2.^2);
Z = -1+2*(x1.^2+x2.^2);
x3=[x1;x2;X;Y;Z];
% Z = sort(Z);
end

%% forth try
function [theta,phi,X,Y,Z] = sample4(n)
u = linspace(0,1,n);
theta = 2*pi*u;
phi = acos(1-2*u);
theta = repmat(theta,1,n);
phi = repmat(phi,n,1);
phi = phi(:);
phi = phi';
X = sin(phi) .* cos(theta);
Y = sin(phi) .* sin(theta);
Z = cos(phi);
% temp = X.^2 + Y.^2 + Z.^2;
% temp(temp>=1) = 0;
% temp = logical(temp);
% X = X(temp);
% Y = Y(temp);
% Z = Z(temp);
disp(sum(imag(X)+imag(Y)+imag(Z)))
end

%% fifth try
function [x1,x2,X,Y,Z] = sample5(n,k)
x1 = linspace(-1,1,n);
x2 = linspace(-1,1,n);
x1 = repmat(x1,1,n);
x2 = repmat(x2,n,1);
x2 = x2(:);
x2 = x2';
temp = x1.^2 + x2.^2;
temp(temp==0) = 0.5;
temp(temp>=1) = 0;
temp = logical(temp);
x1 = x1(temp);
x2 = x2(temp);
X = 2*x1.*sqrt(1-x1.^2-x2.^2);
Y = 2*x2.*sqrt(1-x1.^2-x2.^2);
Z = -1+2*(x1.^2+x2.^2);
%temp = Z;
%temp(temp>=0) = 0;
%temp = logical(temp);
%X = X(temp);
%Y = Y(temp);
%Z = Z(temp);
% Z = sort(Z);
Z = k*Z;
end

%% sixth trt
function [] = sample6()
x1 = linspace(-1,1,n);
x2 = linspace(-1,1,n);
x3 = linspace(-1,1,n);
x4 = linspace(-1,1,n);
x1 = repmat(x1,1,n);
x2 = repmat(x2,n,1);
temp = x1.^2 + x2.^2 + x3.^2 + x4.^2;
temp(temp==0) = 0.5;
temp(temp>=1) = 0;
temp = logical(temp);
x1 = x1(temp);
x2 = x2(temp);
x3 = x3(temp);
x4 = x4(temp);
end
