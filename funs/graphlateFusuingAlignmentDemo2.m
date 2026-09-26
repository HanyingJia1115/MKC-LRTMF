clear
clc
warning off;

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
DataName{11} = 'Flower17';
DataName{12} = 'mfeat';
DataName{13} = 'AR10P';
DataName{14} = 'bbcsport2view';
DataName{15} = 'flower102';
DataName{16} = 'proteinfold';
DataName{17} = 'UCI_DIGIT';
DataName{18} = 'nonpl';
DataName{19} = 'CCV';

datapath = 'E:\桌面常用\论文\论文思路P1\第一篇代码\graph_latefusionalignment';
path = './';
addpath(genpath(path));

for i = 17
    dataName = DataName{i};

    load([datapath,'\datasets\',dataName,'_Kmatrix'],'KH','Y');

    numclass = length(unique(Y));
    numker = size(KH,3);
    num = size(KH,1);

    KH = kcenter(KH);
    KH = knorm(KH);

    %% ========== 这里是你要改的唯一地方 ========== %%
    alpha  = 2;     % ← 自己设
    beta    = 2;     % ← 自己设
    gamma   = 0.01;     % ← 自己设
    %% ========================================== %%

    fprintf('===== 单参数实验: λ=%.4f, β=%.4f, γ=%.4f =====\n',alpha,beta,gamma);

    tic
    [F,iter,obj,AA] = graphlatefusionalignmentclustering2(...
        KH,numclass,alpha,beta,gamma,Y);
    single_time = toc;

    %% ===== 画收敛曲线 =====
    figure;
    x = 1:length(obj);
    set(gca,'FontName','Times New Roman','FontSize',10);
    plot(x, obj,'-o','LineWidth',1.5);
%     title(['Objective Curve (',dataName,')']);
title(['Objective Curve (',dataName,')'],'Interpreter','none');    xlabel('Number of Iterations');
    ylabel('Objective value');
    grid on;

    %% ===== 计算聚类指标 =====
    res = myNMIACCwithmean(F,Y,numclass);
    ACC = res(1); 
    NMI = res(2); 
    Purity = res(3);

    fprintf('\n===== 实验结果 =====\n');
    fprintf('ACC=%.4f, NMI=%.4f, Purity=%.4f\n',ACC,NMI,Purity);
    fprintf('迭代次数=%d\n',iter);
    fprintf('耗时=%.2f 秒\n',single_time);

end
