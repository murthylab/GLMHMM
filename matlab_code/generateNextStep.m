function [newstim,newemit] = generateNextStep(stim,emit_w,numstates,numbins)
    T = size(stim,2);
    newstim = stim;
    filtpower = reshape(sum(reshape(repmat(emit_w,[1,1,T]),[numstates,numbins,T]) .* repmat(reshape(stim(:,:),1,numbins,T),[numstates,1,1]),2),numstates,T);
    p = [ones(1,T); exp(filtpower)] ./ repmat(1+sum(exp(filtpower),1),[size(emit_w,1)+1,1]);

    newemit = sum(repmat(rand(1,T),[size(emit_w,1)+1,1]) < cumsum(p),1);

    newstim(601,newemit == 1) = max(max(newstim(601:630,:)));
    newstim(601,newemit ~= 1) = min(min(newstim(601:630,:)));

    newstim(571,newemit == 2) = max(max(newstim(571:600,:)));
    newstim(571,newemit ~= 2) = min(min(newstim(571:600,:)));

    newstim(511,newemit == 3) = max(max(newstim(511:540,:)));
    newstim(511,newemit ~= 3) = min(min(newstim(511:540,:)));
end