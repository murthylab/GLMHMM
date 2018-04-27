function [z] = getFrameAtSample(sample,frame2sample)
    z = find(abs(sample - frame2sample) == min(abs(sample - frame2sample)));
    z = z(1);
end