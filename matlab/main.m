close all
figure;
axis equal;
hold on
grid on
xlabel('x-axis')
ylabel('y-axis')
zlabel('z-axis')
% [X,Y,Z]=sample_spherePoint();
X=-1;
Y=-1;
Z=1;
camera = transform_camera;
myinit(camera)
mycallZ(camera,[X;Y;Z])
