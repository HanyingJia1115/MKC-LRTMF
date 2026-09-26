function [Z, pivot_index, actual_rank] = PCD_rpcholesky_kernel(K, target_rank)
%RPCHOLESKY_KERNEL Low-rank approximation of a positive semidefinite kernel.
%   K ≈ Z * Z'

[n, m] = size(K);

if n ~= m
    error('K must be a square matrix.');
end

if target_rank < 1 || target_rank > n
    error('target_rank must be between 1 and size(K,1).');
end

if any(~isfinite(K(:)))
    error('K contains NaN or Inf.');
end

% Remove small numerical asymmetry.
K = (K + K') / 2;

residual_diag = diag(K);
diagonal_scale = max(1, max(abs(residual_diag)));

if min(residual_diag) < -1e-10 * diagonal_scale
    error('K may not be positive semidefinite.');
end

residual_diag = max(residual_diag, 0);
initial_trace = sum(residual_diag);
tolerance = max(1e-14, 1e-12 * initial_trace);

Z = zeros(n, target_rank);
pivot_index = zeros(1, target_rank);
actual_rank = 0;

for t = 1:target_rank
    residual_trace = sum(residual_diag);

    if residual_trace <= tolerance
        break;
    end

    % Choose the point with the largest residual.
    [pivot_value, pivot] = max(residual_diag);

    if pivot_value <= tolerance
        break;
    end

    if actual_rank == 0
        residual_column = K(:, pivot);
    else
        Z_old = Z(:, 1:actual_rank);
        residual_column = K(:, pivot) ...
            - Z_old * Z_old(pivot, :)';
    end

    column_pivot = residual_column(pivot);

    if column_pivot <= 0
        if column_pivot > -1e-10 * diagonal_scale
            column_pivot = pivot_value;
        else
            error('A negative pivot was encountered.');
        end
    end

    new_column = residual_column / sqrt(column_pivot);

    if any(~isfinite(new_column))
        error('The Cholesky factor contains NaN or Inf.');
    end

    actual_rank = actual_rank + 1;
    Z(:, actual_rank) = new_column;
    pivot_index(actual_rank) = pivot;

    residual_diag = residual_diag - abs(new_column).^2;

    if min(residual_diag) < -1e-8 * diagonal_scale
        error('The residual diagonal contains a significant negative value.');
    end

    residual_diag = max(residual_diag, 0);
    residual_diag(pivot) = 0;
end

pivot_index = pivot_index(1:actual_rank);

end