function [symblik] = GLMHMM_SymbLik(emit_w,stim,symb)
    numstates = size(emit_w,1);
    numsymb= size(emit_w,2);
    stim_e = repmat(reshape(stim,1,1,size(stim,1),size(stim,2)),[numstates,numsymb,1,1]);
    symblik = zeros(size(emit_w,1),length(symb));

    for t=1:length(symb)
        symblik(:,t) = 1./ (1 + sum(exp(sum(emit_w.*stim_e(:,:,:,t),3)),2));
        if symb(t) ~= 0
            symblik(:,t) = symblik(:,t) .* exp(sum(emit_w(:,symb(t),:).*stim_e(:,symb(t),:,t),3));
        end
        if any(isnan(symblik(:,t)))
            display('oh dear');
        end
    end
end