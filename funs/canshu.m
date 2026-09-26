clc; clear; close all;

%% ===== 1. 读取你的结果 =====
data = xlsread('YALE.xlsx');

alpha = data(:,1);   % 第一列
beta   = data(:,2);   % 第二列
gamma  = data(:,3);   % 第三列
ACC    = data(:,4);   % 第四列

%% ===== 2. 你的真实搜索集合 =====
lambdaset = 2.^(-1:1:7);
betaset   = 2.^(-1:1:7);
gammaset  = 2.^(-1:1:7);

%% ===== 3. 你要固定的 γ =====
gamma_fix = 1;   % ← 你自己改成 0.5 / 2 / 8 都行

mask = abs(gamma - gamma_fix) < 1e-12;

sub_lambda = alpha(mask);
sub_beta   = beta(mask);
sub_ACC    = ACC(mask);

%% ===== 4. 构建与搜索顺序完全一致的矩阵 =====
A = zeros(length(betaset), length(lambdaset));

for i = 1:length(betaset)
    for j = 1:length(lambdaset)

        idx = abs(sub_beta - betaset(i)) < 1e-12 & ...
              abs(sub_lambda - lambdaset(j)) < 1e-12;

        if any(idx)
            A(i,j) = sub_ACC(idx);
        else
            A(i,j) = NaN;
        end
    end
end

%% ===== 5. 画图 =====
figure;
bar3(A, 0.6);

set(gca,'FontName','Times New Roman','FontSize',10);

xlabel('\alpha');
ylabel('\beta');
zlabel('ACC');

xticks(1:length(lambdaset));
xticklabels(string(lambdaset));

yticks(1:length(betaset));
yticklabels(string(betaset));

title(['\gamma = ', num2str(gamma_fix)]);

colorbar;
axis tight;
