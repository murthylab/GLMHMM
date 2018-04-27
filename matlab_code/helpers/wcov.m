function [ output ] = wcov(x,y,w)
    output = sum(w .* (x - wmean(x,w)) .* (y - wmean(y,w))) ./ sum(w);
end