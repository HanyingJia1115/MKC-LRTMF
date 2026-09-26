function [F,iter,obj,AA] = MKCLRTMF(KH,k,alpha,beta,gamma,Y)
num = size(KH, 1);
numker = size(KH, 3);
maxIter = 100;

miu = 1.2;
rho = 1e-2;
lambda = ones(numker,1) / sqrt(numker);
G = zeros(num,k);
H = zeros(num,k);
F = zeros(num,k);

opt.disp = 0;

flag = 1;
iter = 0;

viewnum = 2;

for iv = 1:viewnum
    A{iv} = zeros(num,k);
    W{iv} = zeros(num,k);
end

R = cell(numker,1);
for iv = 1:numker
    R{iv} = eye(k);
end

if 10*k <= num
    num_anchors = 10*k;
else
    num_anchors = floor(num/2);
end

Z = zeros(num, num_anchors, numker);
rInd_temp = zeros(numker, num_anchors);
actual_rank = zeros(numker, 1);

for i = 1:numker
    current_kernel = KH(:,:,i);
    current_kernel = (current_kernel + current_kernel') / 2;

    [Z(:,:,i), selected_index, actual_rank(i)] = ...
        PCD_rpcholesky_kernel(current_kernel, num_anchors);

    rInd_temp(i, 1:actual_rank(i)) = selected_index;
end

for p = 1:numker
    [Up,~,~] = svd(Z(:,:,p),'econ');

    if actual_rank(p) < k
        error('第%d个核的实际近似秩小于聚类数。', p);
    end

    B{p} = Up(:,1:k);
end

while flag
    iter = iter + 1;

    M = zeros(num,k);
    for i = 1:numker
        Zi = Z(:,:,i);
        M = M + Zi*(Zi'*H);
    end

    M = M + 0.5*(alpha*F + rho*A{1} - W{1});
    [U,~,V] = svd(M,'econ');
    H = U*V';

    LHpRp = zeros(num,k);
    for v = 1:numker
        LHpRp = LHpRp + lambda(v)*B{v}*R{v};
    end

    D = beta*LHpRp + alpha*H + rho*A{2} - W{2};
    [U,~,V] = svd(beta*LHpRp + alpha*H + rho*A{2} - W{2},'econ');
    F = U*V';

    for v = 1:numker
        [Ur,~,Vr] = svd(lambda(v)*B{v}'*F);
        R{v} = Ur*Vr';
    end

    f2 = zeros(1,numker);
    for v = 1:numker
        f2(v) = trace(F'*B{v}*R{v}) + eps;
    end
    lambda = f2./norm(f2,2);

    S_tensor = cat(3,H,F);
    W_tensor = cat(3,W{:,:});
    Sv = S_tensor(:);
    Wv = W_tensor(:);
    [Av,objV] = wshrinkObj(Sv + 1/rho*Wv,gamma/rho,...
        [num,k,viewnum],0,3);
    AA(iter) = objV;
    A_tensor = reshape(Av,[num,k,viewnum]);

    for iv = 1:viewnum
        A{iv} = A_tensor(:,:,iv);
        W{iv} = W{iv} + rho*(S_tensor(:,:,iv) - A{iv});
    end

    clear A_tensor S_tensor W_tensor Av Sv Wv
    rho = min(miu*rho,1e10);

    term1 = 0;
    for i = 1:numker
        Zi = Z(:,:,i);
        term1 = term1 ...
            + norm(Zi,'fro')^2 ...
            - norm(Zi'*H,'fro')^2;
    end

    LHpRp_new = zeros(num,k);
    for v = 1:numker
        LHpRp_new = LHpRp_new + lambda(v)*B{v}*R{v};
    end

    term2 = beta*trace(F'*LHpRp_new);
    term3 = alpha*trace(F'*H);
    term4 = objV;

    obj(iter) = -term1 + term2 + term3 - gamma*term4;

    if iter > 30 && ...
            (abs((obj(iter-1)-obj(iter))/obj(iter-1)) < 1e-5 || iter > maxIter)
        flag = 0;
    end
end
end
