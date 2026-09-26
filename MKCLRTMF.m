function [F,iter,obj,AA] = MKCLRTMF(KH,k,alpha,beta,gamma,Y)
%F是最终的聚类指示矩阵，大小为【num，k】；iter是算法的迭代次数
num = size(KH, 1); %样本数
numker = size(KH, 3); %m视图数
maxIter = 100; %the number of iterations
% H = zeros(num,k,numker);
%%

% B=cell(numker,1);
% R=cell(numker,1);
miu = 1.2; %增长因子
rho = 1e-2;
lambda = ones(numker,1) / sqrt(numker);
% lambda = ones(numker,1)/(numker);
G = zeros(num,k);
H = zeros(num,k);
F = zeros(num,k);

opt.disp = 0;           %设置eigs函数的显示选项为不显示

% %% 从 KH 提取每个核的前 k 个特征向量
% K0 = zeros(num, num);
% 
% for p = 1:numker
% 
%     Kp = real((KH(:,:,p) + KH(:,:,p)') / 2);
%     KH(:,:,p) = Kp;
% 
%     if any(~isfinite(Kp(:)))
%         error('第 %d 个核矩阵包含 NaN 或 Inf。', p);
%     end
% 
%     [Hp, ~] = eigs(Kp, k, 'la', opt);
% 
%     if any(~isfinite(Hp(:)))
%         error('第 %d 个核的特征向量包含 NaN 或 Inf。', p);
%     end
% 
%     K0 = K0 + (1/numker) * Kp;
%     HP(:,:,p) = Hp;
%     B{p} = Hp;
% 
% end

% gamma = 1;
flag = 1;               %设置一个标志位，用于控制循环
iter = 0;


viewnum = 2;

for iv = 1 : viewnum
    A{iv} = zeros(num,k);
    W{iv} = zeros(num,k);

end

% B{iv} is H_p in the formule 

% for iv = 1 : numker
% %     B{iv} = zeros(num,k);
%     R{iv} = zeros(k,k);
% end
R = cell(numker,1);
for iv = 1:numker
    R{iv} = eye(k);
end

% 按照原来的规则设置地标数量
if 10*k <= num
    num_anchors = 10*k;
else
    num_anchors = floor(num/2);
end
%% 使用 RPCholesky 构造低秩核近似
Z = zeros(num, num_anchors, numker);
% apprxK_KKM = zeros(num, num, numker);

rInd_temp = zeros(numker, num_anchors);
actual_rank = zeros(numker, 1);

for i = 1:numker

    current_kernel = KH(:,:,i);

    % 保证核矩阵对称
    current_kernel = real((current_kernel + current_kernel') / 2);

    % RPCholesky 直接返回低秩因子
    [Z(:,:,i), selected_index, actual_rank(i)] = ...
        rpcholesky_kernel(current_kernel, num_anchors);

    % 保存实际选中的地标下标
    rInd_temp(i, 1:actual_rank(i)) = selected_index;

    % 得到近似核矩阵
%     apprxK_KKM(:,:,i) = Z(:,:,i) * Z(:,:,i)';

%     fprintf('第 %d 个核：RPCholesky 实际选择 %d/%d 个地标\n', ...
%         i, actual_rank(i), num_anchors);
end

for p = 1:numker
[Up,~,~] = svd(Z(:,:,p),'econ');

if actual_rank(p) < k
    error('第%d个核的实际近似秩小于聚类数。', p);
end

B{p} = Up(:,1:k);
end

% 使用近似核替换原核矩阵
% KH = apprxK_KKM;

% K0 = zeros(num,num);
% for p = 1:numker
% 
%     KH(:,:,p) = (KH(:,:,p) + KH(:,:,p)') / 2;
% 
%     if actual_rank(p) < k
%         error('第 %d 个核的实际秩 %d 小于聚类数 %d。', ...
%             p, actual_rank(p), k);
%     end
% 
%     [Hp_all, ~, ~] = svd(Z(:,:,p), 'econ');
% 
%     Hp = Hp_all(:, 1:k);
% 
%     K0 = K0 + (1/numker) * KH(:,:,p);
%     HP(:,:,p) = Hp;
%     B{p} = Hp;
% end

% K0 = zeros(num,num);
% for p=1:numker % m - kernels
%     KH(:,:,p) = (KH(:,:,p)+KH(:,:,p)')/2;
%     [Hp, ~] = eigs(KH(:,:,p), k, 'la', opt);
%     K0 = K0 + (1/numker)*KH(:,:,p);
%     HP(:,:,p) = Hp;
%     B{p} = HP(:,:,p);
% end
while flag
    iter = iter +1;

    %% the first step-- optimize H
    M = zeros(num,k);
    for i=1:numker
%         [u,s,v]= svd(Z(:,:,i)*(Z(:,:,i)' * H) + alpha * F + rho *A{1} - W{1},'econ');
%         H = u*v';
        Zi = Z(:,:,i);
        M = M + Zi*(Zi'*H);
       
    end
        M = M + 0.5*(alpha*F + rho*A{1} - W{1});
        [U,~,V] = svd(M,'econ');
        H = U*V';

%% the second step-- optimize F
   LHpRp = zeros(num,k);
   for v=1:numker
      LHpRp = LHpRp + lambda(v)*B{v}*R{v};
   end
   D =beta * LHpRp + alpha * H + rho * A{2} - W{2};
   [U,~,V]= svd(beta * LHpRp + alpha * H + rho * A{2} - W{2},'econ');
   F = U*V';
     %% update R^v
     for v=1:numker
        [Ur,~,Vr]=svd(lambda(v)*B{v}'*F);
        R{v}=Ur*Vr';
     end
    %% update lambda
       f2 = zeros(1,numker);
       for v=1:numker
            f2(v) = trace(F' * B{v} * R{v}) + eps; 
       end
      lambda = f2./norm(f2,2);
   
%% == update A{i} ==
    S_tensor = cat(3, H, F); 
    W_tensor = cat(3, W{:,:});
    Sv = S_tensor(:);
    Wv = W_tensor(:);
    [Av, objV] = wshrinkObj(Sv + 1/rho*Wv,gamma/rho,[num,k,viewnum],0,3);
    AA(iter)  = objV;
    A_tensor = reshape(Av, [num,k,viewnum]);
    for iv = 1:viewnum
        A{iv} = A_tensor(:,:,iv);
        W{iv} = W{iv}+rho*(S_tensor(:,:,iv)-A{iv});
    end 
       clear A_tensor S_tensor W_tensor Av Sv Wv  
    rho = min(miu*rho, 1e10);
    %end


    term1 =0;
    for i = 1 : numker
%         term1 = term1 + trace(KH(:,:,i)*(eye(num)-H*H'));
Zi = Z(:,:,i);

term1 = term1 ...
    + norm(Zi,'fro')^2 ...
    - norm(Zi'*H,'fro')^2;
   
    end

   LHpRp_new = zeros(num,k);
   for v=1:numker
      LHpRp_new = LHpRp_new + lambda(v)*B{v}*R{v};
   end
   term2 = beta*trace(F'*LHpRp_new);
   term3 = alpha*trace(F'*H);
   term4 = objV;

    obj(iter) = (-term1) + term2+ term3 - gamma*term4;
    
     if (iter>30) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-5 || iter>maxIter)
        flag =0;
     end
% a = 1;
end
