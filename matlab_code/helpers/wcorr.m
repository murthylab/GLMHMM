function [ output ] = wcorr(x,y,w)
    output = wcov(x,y,w) ./ sqrt(wcov(x,x,w) * wcov(y,y,w));
end