function [Z, pivot_index, actual_rank] = rpcholesky_kernel(K, target_rank)
%RPCHOLESKY_KERNEL 随机主元 Cholesky 核矩阵近似
%
% 输入：
%   K           n×n 的半正定核矩阵
%   target_rank 目标近似秩，也就是地标数量
%
% 输出：
%   Z           n×target_rank 的低秩因子，满足 K ≈ Z*Z'
%   pivot_index 被选中的地标样本下标
%   actual_rank 实际得到的近似秩
%
% 地标采样概率：
%   probability(i) = residual_diagonal(i) / sum(residual_diagonal)

    [n, m] = size(K);

    if n ~= m
        error('输入的核矩阵 K 必须是方阵。');
    end

    if target_rank < 1 || target_rank > n
        error('target_rank 必须满足 1 <= target_rank <= size(K,1)。');
    end

    if any(~isfinite(K(:)))
        error('输入核矩阵包含 NaN 或 Inf。');
    end

    % 保证核矩阵严格对称
    K = (K + K') / 2;

    % 初始残差矩阵就是 K，因此残差对角线为 diag(K)
    residual_diag = diag(K);

    diagonal_scale = max(1, max(abs(residual_diag)));

    % RPCholesky 要求 K 是半正定矩阵
    if min(residual_diag) < -1e-10 * diagonal_scale
        error(['核矩阵对角线上存在明显负数，矩阵可能不是半正定矩阵。' ...
               '最小对角元素为 %.6e。'], min(residual_diag));
    end

    % 消除浮点误差造成的极小负数
    residual_diag = max(residual_diag, 0);

    initial_trace = sum(residual_diag);
    stop_tolerance = max(1e-14, 1e-12 * initial_trace);

    % 即使提前停止，也保持 Z 的列数等于 target_rank
    Z = zeros(n, target_rank);
    pivot_index = zeros(1, target_rank);

    actual_rank = 0;

    for t = 1:target_rank

        residual_trace = sum(residual_diag);

        % 剩余误差已经非常小，提前停止
        if residual_trace <= stop_tolerance
            break;
        end

        %% 按照残差对角线进行随机采样
%         random_value = rand() * residual_trace;
%         cumulative_value = cumsum(residual_diag);
% 
%         pivot = find(cumulative_value > random_value, 1, 'first');

        [~, pivot] = max(residual_diag);

        % 处理极少数浮点数舍入情况
        if isempty(pivot)
            pivot = find(residual_diag > 0, 1, 'last');
        end

        pivot_value = residual_diag(pivot);

        if pivot_value <= stop_tolerance
            residual_diag(pivot) = 0;
            continue;
        end

        %% 计算当前残差矩阵的第 pivot 列
        if actual_rank == 0
            residual_column = K(:, pivot);
        else
            Z_old = Z(:, 1:actual_rank);

            residual_column = K(:, pivot) ...
                - Z_old * Z_old(pivot, :)';
        end

        % 理论上 residual_column(pivot) 等于 pivot_value
        column_pivot_value = real(residual_column(pivot));

        if column_pivot_value <= 0
            % 如果只是微小数值误差，使用维护的残差对角元素
            if column_pivot_value > -1e-10 * diagonal_scale
                column_pivot_value = pivot_value;
            else
                error(['RPCholesky 遇到负主元 %.6e，' ...
                       '输入核矩阵可能不是半正定矩阵。'], ...
                       column_pivot_value);
            end
        end

        %% 生成新的 Cholesky 列
        new_column = residual_column / sqrt(column_pivot_value);

        if any(~isfinite(new_column))
            error('RPCholesky 产生了 NaN 或 Inf，请检查核矩阵。');
        end

        actual_rank = actual_rank + 1;
        Z(:, actual_rank) = new_column;
        pivot_index(actual_rank) = pivot;

        %% 更新残差对角线
        new_residual_diag = residual_diag - abs(new_column).^2;

        % 明显负数通常说明核矩阵不满足半正定条件
        if min(new_residual_diag) < -1e-8 * diagonal_scale
            error(['RPCholesky 残差出现明显负值 %.6e，' ...
                   '核矩阵可能不是半正定矩阵。'], ...
                   min(new_residual_diag));
        end

        residual_diag = max(real(new_residual_diag), 0);

        % 已经选择的样本不能重复选择
        residual_diag(pivot) = 0;
    end

    % 删除地标下标中未使用的零元素
    pivot_index = pivot_index(1:actual_rank);
end