function [output] = wvar(x, w)
    output = sum(w.* (x - wmean(x,w)).^2) ./ sum(w);
end