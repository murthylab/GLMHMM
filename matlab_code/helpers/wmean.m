function [output] = wmean(x,w)
    output = sum(w.*x)./sum(w);
end