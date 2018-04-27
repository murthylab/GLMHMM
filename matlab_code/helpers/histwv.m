function [histw histv] = histwv(v, w, mn, mx, bins) 
%Inputs: 
%vv - values 
%ww - weights 
%minV - minimum value 
%maxV - max value 
%bins - number of bins (inclusive) 

%Outputs: 
%histw - wieghted histogram 
%histv (optional) - histogram of values 

delta = (mx-mn)/(bins-1); 
subs = round((v-mn)/delta)+1; 

histv = accumarray(subs,1,[bins,1]); 
histw = accumarray(subs,w,[bins,1]); 
end