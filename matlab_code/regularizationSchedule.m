function [lambda] = regularizationSchedule(iter)
    tau = 1.5;

    % schedule 1: by iteration
    if iter < 5
        lambda = 1;
    else
        lambda = exp(-(iter-4) / tau) + 0.005;
    end

end