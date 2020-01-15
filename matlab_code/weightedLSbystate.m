function [outweights,outstd] = weightedLSbystate(thesestim, thesesymb, thesegamma, tsize)
    numFits = 10;
    numSubData = 0.3;
    numbins = size(thesestim,2);
    numstates = size(thesegamma,1);

    outweights = zeros(numstates,numbins);
    outstd = zeros(numstates,numbins);

    for states = 1:numstates
        tik = diag(ones(numbins,1))*tsize^2;

        nweights = zeros(numFits,numbins);
        for nrepeats=1:numFits
            usestim = randperm(size(thesestim,1));
            usestim = usestim(1:round(length(usestim)*numSubData));
            W = diag(thesegamma(states,usestim));

            Z = (thesestim(usestim,:)'*W*thesestim(usestim,:) + tik)^-1;
            STA = thesestim(usestim,:)'*W*thesesymb(usestim);
            nweights(nrepeats,:) = Z * STA;
        end
        outweights(states,:) = mean(nweights);
        outstd(states,:) = std(nweights);
    end
end