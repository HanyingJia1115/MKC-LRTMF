clear
clc
warning off

DataName{1} = 'caltech101_nTrain5_48';
DataName{2} = 'caltech101_nTrain10_48';
DataName{3} = 'caltech101_nTrain15_48';
DataName{4} = 'caltech101_nTrain20_48';
DataName{5} = 'caltech101_nTrain25_48';
DataName{6} = 'caltech101_nTrain30_48';
DataName{7} = 'CCV';
DataName{8} = 'ORL';
DataName{9} = 'flower17_numberOf_170';
DataName{10} = 'YALE';
DataName{11} = 'plant';
DataName{12} = 'mfeat';
DataName{13} = 'AR10P';
DataName{14} = 'bbcsport2view';
DataName{15} = 'flower102';
DataName{16} = 'proteinfold';
DataName{17} = 'UCI_DIGIT';
DataName{18} = 'nonpl';
DataName{19} = 'CCV';

datapath = pwd;
addpath(genpath(datapath));

dataset_indices = 13;

for i = dataset_indices

    dataName = DataName{i};
    load([datapath, '\datasets\', dataName, '_Kmatrix'], 'KH', 'Y');

    numclass = length(unique(Y));

    KH = kcenter(KH);
    KH = knorm(KH);

    %% Parameters
    alphaset = 2.^(-1:1:7);
    betaset  = 2.^(-1:1:7);
    gammaset = 2.^(-1:1:7);

    Temp = -inf;
    best_res = zeros(1, 10);

    total_exp = length(alphaset) * length(betaset) * length(gammaset);
    exp_count = 0;

    % alpha beta gamma ACC NMI Purity ACC_STD NMI_STD Purity_STD Time
    AllResults = zeros(total_exp, 10);

    total_tic = tic;

    for lam1 = 1:length(alphaset)
        for lam2 = 1:length(betaset)
            for lam3 = 1:length(gammaset)

                exp_count = exp_count + 1;

                alpha = alphaset(lam1);
                beta  = betaset(lam2);
                gamma = gammaset(lam3);

                single_tic = tic;

                [F,iter,obj,AA] = MKCLRTMF( ...
                    KH, numclass, alpha, beta, gamma, Y);

                single_time = toc(single_tic);

                res = myNMIACCwithmean(F, Y, numclass);

                ACC = res(1);
                NMI = res(2);
                Purity = res(3);

                ACC_STD = res(4);
                NMI_STD = res(5);
                Purity_STD = res(6);

                current_res = [alpha, beta, gamma, ...
                    ACC, NMI, Purity, ...
                    ACC_STD, NMI_STD, Purity_STD, single_time];

                AllResults(exp_count, :) = current_res;

                if ACC > Temp
                    Temp = ACC;
                    best_res = current_res;
                end

                fprintf(['Exp %d/%d | ' ...
                    'ACC=%.4f ± %.4f | ' ...
                    'NMI=%.4f ± %.4f | ' ...
                    'Purity=%.4f ± %.4f | ' ...
                    'alpha=%.4f beta=%.4f gamma=%.4f | ' ...
                    'BestACC=%.4f | Time=%.2fs\n'], ...
                    exp_count, total_exp, ...
                    ACC, ACC_STD, ...
                    NMI, NMI_STD, ...
                    Purity, Purity_STD, ...
                    alpha, beta, gamma, Temp, single_time);

            end
        end
    end

    total_time = toc(total_tic);

    fprintf('\n===================== %s =====================\n', dataName);

    fprintf('Total time: %.2f s (%.2f min)\n', ...
        total_time, total_time / 60);

    fprintf('\nBest parameters:\n');
    fprintf('alpha=%.4f beta=%.4f gamma=%.4f\n', ...
        best_res(1), best_res(2), best_res(3));

    fprintf('ACC=%.4f ± %.4f\n', ...
        best_res(4), best_res(7));

    fprintf('NMI=%.4f ± %.4f\n', ...
        best_res(5), best_res(8));

    fprintf('Purity=%.4f ± %.4f\n', ...
        best_res(6), best_res(9));

    fprintf('Time=%.2f s\n', best_res(10));

end