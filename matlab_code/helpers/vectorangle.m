function [angle] = vectorangle(u,v) 
    angle = acosd(dot(u,v)/(sqrt(sum(u.^2))*sqrt(sum(v.^2))));
end