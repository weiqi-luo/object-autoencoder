classdef transform_camera
    properties    
        oo = [0,0,0,1]';
        ox = [1,0,0,1]';
        oy = [0,1,0,1]';
        oz = [0,0,1,1]';
    end
    methods
        function myinit(obj)
            obj.draw(obj.oo,obj.ox,obj.oy,obj.oz,'b');
        end
        
        function mycall(obj,p)
            for i = 1:size(p,2)
                q = p(:,i);
                plot3([0;q(1)],[0;q(2)],[0;q(3)],"-o")
                theta_z = atan(q(2)/q(1))+heaviside(q(1))*pi;
                theta_y = asin(q(3)/sqrt(q(1).^2+q(2).^2+q(3).^2));
                [oo,ox,oy,oz] = obj.transformation ...
                    (obj.oo,obj.ox,obj.oy,obj.oz,q,0,theta_y,theta_z);
                obj.draw(oo,ox,oy,oz,'r');
            end
        end
        
                function mycallZ(obj,p)
            for i = 1:size(p,2)
                q = p(:,i);
                plot3([0;q(1)],[0;q(2)],[0;q(3)],"-o")
                theta_y = atan(q(1)/q(3))+heaviside(q(3))*pi;
                theta_x = asin(q(2)/sqrt(q(1).^2+q(2).^2+q(3).^2));
                [oo,ox,oy,oz] = obj.transformation ...
                    (obj.oo,obj.ox,obj.oy,obj.oz,q,theta_x,theta_y,0);
                obj.draw(oo,ox,oy,oz,'r');
            end
        end
        
        function [oo,ox,oy,oz] = transformation(obj,oo,ox,oy,oz,p,theta_x,theta_y,theta_z) 
            Rx = [1 0 0 0; 
                0 cos(theta_x) -sin(theta_x) 0; 
                0 sin(theta_x) cos(theta_x) 0;
                0 0 0 1];
            Ry = [cos(theta_y) 0 sin(theta_y) 0; 
                0 1 0 0; 
                -sin(theta_y) 0 cos(theta_y) 0
                0 0 0 1];
            Rz = [cos(theta_z) -sin(theta_z) 0 0; 
                sin(theta_z) cos(theta_z) 0 0; 
                0 0 1 0
                0 0 0 1];
            T = [1 0 0 p(1,:); 
                 0 1 0 p(2,:); 
                 0 0 1 p(3,:); 
                 0 0 0 1]*Rz*Ry*Rx;
            oo = T*oo;
            ox = T*ox;
            oy = T*oy;
            oz = T*oz;
        end
        
        function draw(obj,oo,ox,oy,oz,color)
            x = [oo(1)  oo(1)  oo(1);  ox(1)  oy(1)  oz(1)];
            y = [oo(2)  oo(2)  oo(2);  ox(2)  oy(2)  oz(2)];
            z = [oo(3)  oo(3)  oo(3);  ox(3)  oy(3)  oz(3)];
%             x = [oo(1) ;  ox(1)];
%             y = [oo(2) ;  ox(2)];
%             z = [oo(3) ;  ox(3)];
            plot3(x,y,z,strcat(color,'-*'))
            text(oo(1),oo(2),oo(3),strcat(color,'-o'))
            text(ox(1),ox(2),ox(3),"x")
            text(oy(1),oy(2),oy(3),"y")
            text(oz(1),oz(2),oz(3),"z")
            axis([-2 2    -2 2    -2 2])
        end
    end
end

