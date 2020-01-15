function [thesestim,thesesymb,thesegamma] = collectWLSInfo(stim)
    % I should compute the sigma of the fit for later use as well

    numstates = stim{1}.numstates;
    numbins = stim{1}.numtotalbins;

    totalemit = 0;
    for trial=1:length(stim)
        totalemit = totalemit + sum(stim{trial}.goodemit);
    end

    thesegamma = zeros(numstates,totalemit,1);
    thesesymb = zeros(totalemit,1);
    thesestim= zeros(totalemit,numbins);
    ind = 0;
    for trial = 1:length(stim)
        goodemit = stim{trial}.goodemit;
        T = sum(goodemit);

        thesegamma(:,ind + (1:T)) = stim{trial}.gamma(:,goodemit);
        thesestim(ind + (1:T),:) = stim{trial}.data(:,goodemit)';
        thesesymb(ind + (1:T)) = stim{trial}.symb(goodemit);

        ind = ind + T;
    end
end
