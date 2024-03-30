function in = localResetFcn(in)

    % Randomize reference signal
    blk = sprintf("rlwatertank/Desired \nWater Level");
    h = 3*randn + 10;
    while h <= 0 || h >= 20
        h = 3*randn + 10;
    end
    in = setBlockParameter(in,blk,Value=num2str(h));

    % Randomize initial height
    h = 3*randn + 10;
    while h <= 0 || h >= 20
        h = 3*randn + 10;
    end
    blk = "rlwatertank/Water-Tank System/H";
    in = setBlockParameter(in,blk,InitialCondition=num2str(h));

end