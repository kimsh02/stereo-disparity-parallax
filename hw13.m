function [q] = hw13()
% usage:
%  q = hw13()
% where q is the number of questions answered
% 
%  assumes all required files for hw13 exist in the same directory as the
%  script

format compact
close all
q = 2;

% A



% Load images
left1 = imread('left-1.tiff');
right1 = imread('right-1.tiff');
left2 = imread('left-2.tiff');
right2 = imread('right-2.tiff');

function disparity_map = compute_disparity(left_img, right_img, D, C)
    [rows, cols] = size(left_img);
    margin = D + C;
    disparity_map = zeros(rows, cols);
    for y = 1:rows
        for x = (1+margin):(cols-margin)
            best_match = 0;
            best_disparity = 0;
            left_vector = double(left_img(y, (x-C):(x+C)));
            if sum(left_vector) > 0
                left_vector = left_vector / norm(left_vector);
            end
    
            for d = -D:D
                right_x = x + d;
                % if right_x - C < 1 || right_x + C > cols
                %     continue;
                % end
                right_vector = double(right_img(y, (right_x-C):(right_x+C)));
                if sum(right_vector) > 0
                    right_vector = right_vector / norm(right_vector);
                end
                dot_product = sum(left_vector .* right_vector);
                if dot_product > best_match
                    best_match = dot_product;
                    best_disparity = abs(d);
                end
            end
            disparity_map(y,x) = best_disparity;
        end
    end
end

if false

disparity_map1 = compute_disparity(left1, right1, 15, 10);
disparity_map2 = compute_disparity(left2, right2, 15, 10);

figure;
subplot(1, 2, 1); imagesc(disparity_map1); colormap gray; title('Disparity Map 1');
subplot(1, 2, 2); imagesc(disparity_map2); colormap gray; title('Disparity Map 2');
saveas(gcf, 'a1.png');

end

function avg_disparity = compute_average_largest_disparities(disparity_map)
    sorted_disparities = sort(disparity_map(:), 'descend');
    top_10_percent = sorted_disparities(1:round(0.1*numel(sorted_disparities)));
    avg_disparity = mean(top_10_percent);
end

if false

avg_disparity1 = compute_average_largest_disparities(disparity_map1);
avg_disparity2 = compute_average_largest_disparities(disparity_map2);
disp(['Average disparity (set 1): ', num2str(avg_disparity1)]);
disp(['Average disparity (set 2): ', num2str(avg_disparity2)]);

b = 10;
f = 2;
pixel_size = 0.025;
z1 = (b * f) / (avg_disparity1 * pixel_size);
z2 = (b * f) / (avg_disparity2 * pixel_size);
% z1 = ((b - (avg_disparity1 * pixel_size)) * f) / (avg_disparity1 * pixel_size);
% z2 = ((b - (avg_disparity2 * pixel_size)) * f) / (avg_disparity2 * pixel_size);
disp(['Estimated distance (set 1): ', num2str(z1), ' cm']);
disp(['Estimated distance (set 2): ', num2str(z2), ' cm']);

end

% B

meters_per_mile = 1609.344
f = 2
b = double(8e7*meters_per_mile)
D = 6.2e-6
z_meters = (b*f) / (D)
z_miles = z_meters / meters_per_mile
lightyears_to_miles = 5.879e12
z_lightyears = z_miles / lightyears_to_miles

end



% % A
% 
% if false
% 
% line_data = importdata('line_data_2.txt');
% fig = figure;
% scatter(line_data(:,1), line_data(:,2));
% title('Scatter Plot With RANSAC Line');
% xlabel('X');
% ylabel('Y');
% hold on;
% 
% % formula for k, 99.99% finding correct fit
% num_iterations = round(log(1-0.9999)/log(1-0.25^(2)))
% best_error = inf;
% % top N%
% inlier_percentage = 0.25;
% 
% for i = 1:num_iterations
%     indices = randperm(size(line_data, 1), 2);
%     p1 = line_data(indices(1), :);
%     p2 = line_data(indices(2), :);
%     a = p2(2) - p1(2);
%     b = p1(1) - p2(1);
%     c = p2(1) * p1(2) - p2(2) * p1(1);
%     distances = abs(a * line_data(:,1) + b * line_data(:,2) + c) / sqrt(a^2 + b^2);
%     [~, sorted_indices] = sort(distances);
%     inlier_count = round(size(line_data, 1) * inlier_percentage);
%     inliers = line_data(sorted_indices(1:inlier_count), :);
%     A = [inliers(:,1) - mean(inliers(:,1)), inliers(:,2) - mean(inliers(:,2))];
%     AtA = A' * A;
%     [eigenvec, eigenval] = eig(AtA);
%     [~, idx] = min(diag(eigenval));
%     a = eigenvec(1, idx);
%     b = eigenvec(2, idx);
%     d = a * mean(inliers(:,1)) + b * mean(inliers(:,2));
%     % perpendicular distance squared
%     error = sum((abs(a*inliers(:,1)+b*inliers(:,2)-d)/sqrt(a^2 + b^2)).^2);
%     if error < best_error
%         best_error = error;
%         slope_intercept = [-a/b, d/b];
%     end
% end
% best_error
% slope_intercept
% best_line = slope_intercept(1) * line_data(:,1) + slope_intercept(2);
% plot(line_data(:,1), best_line, 'r', 'LineWidth', 3);
% legend('', 'RANSAC Best Fit Line', 'FontSize', 14);
% saveas(fig, 'a1.png');
% 
% end
% 
% % B
% 
% if false
% 
% num_samples = 10;
% pairs_4 = cell(1, num_samples);
% pairs_5 = cell(1, num_samples);
% pairs_6 = cell(1, num_samples);
% for i = 1:num_samples
%     pairs_4{i} = [rand(4,2), ones(4,1), rand(4,2)];
%     pairs_5{i} = [rand(5,2), ones(5,1), rand(5,2)];
%     pairs_6{i} = [rand(6,2), ones(6,1), rand(6,2)];
% end
% samples = {pairs_4, pairs_5, pairs_6};
% RMSE_samples = zeros(1,3);
% 
% for i = 1:size(samples, 2)
%     sample = samples{i};
%     fprintf("%d\n", i);
%     for j = 1:num_samples
%         pairs = sample{j};
%         A = pairs(:, 1:3);
%         B = pairs(:, 4:5);
%         U = zeros(3*size(pairs,1), 9);
%         for k = 1:3:3*size(pairs,1)
%             z = zeros(1, 3);
%             idx = ceil(k/3);
%             X = A(idx,:);
%             x_prime = B(idx,1);
%             y_prime = B(idx,2);
%             U(k:k+2, :) = [z, -X, y_prime*X; 
%                 X, z, -x_prime*X;
%                 -y_prime*X, x_prime*X, z];
%         end
%         UtU = U'*U;
%         [eigvec, eigval] = eig(UtU);
%         [~, min_idx] = min(diag(eigval));
%         H = reshape(eigvec(:, min_idx), 3, 3)';
%         homography = (H*A');
%         lambda = homography(3,:);
%         X_prime = [homography(1,:)./lambda; homography(2,:)./lambda]';
%         RMSE_pair = sqrt(mean(sum((X_prime - B).^2, 2)));
%         RMSE_pair
%     end
%     RMSE_samples(1,i) = sqrt(mean(RMSE_pair.^2));
% end
% RMSE_samples
% end
% 
% f1 = imread('frame1.jpg');
% f2 = imread('frame2.jpg');
% f3 = imread('frame3.jpg');
% s1 = imread('slide1.jpeg');
% s2 = imread('slide2.jpeg');
% s3 = imread('slide3.jpeg');
% % figure;datacursormode on;
% % imagesc(s1);
% % figure;datacursormode on;
% % imagesc(f1);
% % figure;datacursormode on;
% % imagesc(s2);
% % figure;datacursormode on;
% % imagesc(f2);
% % figure;datacursormode on;
% % imagesc(s3);
% % figure;datacursormode on;
% % imagesc(f3);
% C1 = importdata('1.txt');
% C1(:, [1,2,4,5]) = C1(:, [2,1,5,4]);
% C2 = importdata('2.txt');
% C2(:, [1,2,4,5]) = C2(:, [2,1,5,4]);
% C3 = importdata('3.txt');
% C3(:, [1,2,4,5]) = C3(:, [2,1,5,4]);
% a1 = C1(:, 1:3);
% b1 = C1(:, 4:5);
% a2 = C2(:, 1:3);
% b2 = C2(:, 4:5);
% a3 = C3(:, 1:3);
% b3 = C3(:, 4:5);
% imgs = {s1, f1, s2, f2, s3, f3};
% clicked = {a1, b1, a2, b2, a3, b3};
% Xsf = {a1(1:2:8,:), a2(1:2:8,:), a3(1:2:8,:)};
% X_primesf = {b1(1:2:8,:), b2(1:2:8,:), b3(1:2:8,:)};
% projected = {};
% H_mat = {};
% 
% for i = 1:size(Xsf, 2)
%     A = Xsf{i};
%     B = X_primesf{i};
%     U = zeros(3*size(Xsf{i},1), 9);
%     for k = 1:3:3*size(Xsf{i},1)
%         z = zeros(1, 3);
%         idx = ceil(k/3);
%         X = A(idx,:);
%         x_prime = B(idx,1);
%         y_prime = B(idx,2);
%         U(k:k+2, :) = [z, -X, y_prime*X; 
%             X, z, -x_prime*X;
%             -y_prime*X, x_prime*X, z];
%     end
%     UtU = U'*U;
%     [eigvec, eigval] = eig(UtU);
%     [~, min_idx] = min(diag(eigval));
%     H = reshape(eigvec(:, min_idx), 3, 3)';
%     H_mat{end+1} = H;
%     homography = (H*A');
%     lambda = homography(3,:);
%     X_prime = [homography(1,:)./lambda; homography(2,:)./lambda]';
%     projected{end+1} = X_prime;
% end
% % for i = 1:size(projected, 2)
% %     projected{i}
% %     X_primesf{i}
% %     H_mat{i}
% % end
% 
% if false
% 
% for i = 1:size(imgs, 2)
% % for i = 3:4
%     fig = figure;
%     imshow(imgs{i});
%     hold on;
%     plot(clicked{i}(1:2:8,1), clicked{i}(1:2:8,2), 'r*', 'MarkerSize', 15);
%     plot(clicked{i}(2:2:8,1), clicked{i}(2:2:8,2), 'c*', 'MarkerSize', 15);
%     if mod(i, 2) == 0
%         homography = H_mat{i/2} * clicked{i-1}';
%         lambda = homography(3,:);
%         X_prime = [homography(1,:)./lambda; homography(2,:)./lambda]';
%         % size(X_prime)
%         % size(clicked{i-1})
%         % size(clicked{i})
%         % X_prime
%         % clicked{i}
%         plot(X_prime(:,1), X_prime(:,2), 'g.', 'MarkerSize', 15);
%     end
%     set(gca,'Position', [0.05 0.05 0.9 0.9]);
%     saveas(fig, sprintf('b%d.png', i));
% end
% 
% end
% 
% % C
% 
% s1s = load("sift_files/slide1.sift");
% s2s = load("sift_files/slide2.sift");
% s3s = load("sift_files/slide3.sift");
% f1s = load("sift_files/frame1.sift");
% f2s = load("sift_files/frame2.sift");
% f3s = load("sift_files/frame3.sift");
% A = zeros(size(s3, 1), size(s3, 2), 3, 'uint8');
% for i = 1:size(A, 1)
%     for j = 1:size(A, 2)
%         g = s3(i,j);
%         A(i,j,:) = [g, g, g];
%     end
% end
% s3 = A;
% 
% nth = 1;
% kk = 1;
% [knn1, d1] = knnsearch(f1s(1:nth:end,5:end), s1s(:,5:end), 'K', kk);
% [knn2, d2] = knnsearch(f2s(1:nth:end,5:end), s2s(:,5:end), 'K', kk);
% [knn3, d3] = knnsearch(f3s(1:nth:end,5:end), s3s(:,5:end), 'K', kk);
% 
% scollage = {s1s, s2s, s3s};
% fcollage = {f1s, f2s, f3s};
% knn = {knn1, knn2, knn3};
% s_img = {s1, s2, s3};
% f_img = {f1, f2, f3};
% for j = 1:size(scollage, 2)
%     S = scollage{j};
%     F = fcollage{j};
%     K = knn{j};
%     S_IMG = s_img{j};
%     F_IMG = f_img{j};
%     % formula for k, 99.9999% finding correct fit
%     % top N%
%     inlier_percentage = 0.08;
%     num_iterations = round(log(1-0.999999)/log(1-inlier_percentage^(4)))
%     best_error = inf;
%     for i = 1:num_iterations
%         % no repeat points
%         while true
%             indices = randperm(size(S, 1), 4)';
%             frame_points = F(K(indices,:),1:2);
%             [unique_frame_points, ~, ~] = unique(frame_points, 'rows');
%             if size(unique_frame_points, 1) == size(frame_points, 1)
%                 slide_points = [S(indices,1:2), ones(size(indices,1), 1)];
%                 break;
%             end
%         end
%         % slide_points
%         % frame_points
%         U = zeros(3*size(slide_points,1), 9);
%         for k = 1:3:3*size(slide_points,1)
%             z = zeros(1, 3);
%             idx = ceil(k/3);
%             X = slide_points(idx,:);
%             x_prime = frame_points(idx,1);
%             y_prime = frame_points(idx,2);
%             U(k:k+2, :) = [z, -X, y_prime*X;
%                 X, z, -x_prime*X;
%                 -y_prime*X, x_prime*X, z];
%         end
%         UtU = U'*U;
%         [eigvec, eigval] = eig(UtU);
%         [~, min_idx] = min(diag(eigval));
%         H = reshape(eigvec(:, min_idx), 3, 3)';
%         homography = (H*[S(:,1:2), ones(size(S, 1), 1)]');
%         lambda = homography(3,:);
%         X_prime = [homography(1,:)./lambda; homography(2,:)./lambda]';
%         SE = (sum((X_prime - F(K(:,1),1:2)).^2, 2));
%         % RMSE = sqrt(mean(sum((X_prime - F(K(:,1),1:2)).^2, 2)))
%         [~, sorted_indices] = sort(SE);
%         inlier_count = round(size(SE, 1) * inlier_percentage);
%         slide_inliers = [S(sorted_indices(1:inlier_count,1), 1:2), ones(inlier_count, 1)];
%         frame_inliers = F(K(sorted_indices(1:inlier_count,1),1), 1:2);
%         % size(slide_inliers)
%         % size(frame_inliers)
%         U = zeros(3*size(slide_inliers,1), 9);
%         for k = 1:3:3*size(slide_inliers,1)
%             z = zeros(1, 3);
%             idx = ceil(k/3);
%             X = slide_inliers(idx,:);
%             x_prime = frame_inliers(idx,1);
%             y_prime = frame_inliers(idx,2);
%             U(k:k+2, :) = [z, -X, y_prime*X;
%                 X, z, -x_prime*X;
%                 -y_prime*X, x_prime*X, z];
%         end
%         UtU = U'*U;
%         [eigvec, eigval] = eig(UtU);
%         [~, min_idx] = min(diag(eigval));
%         H = reshape(eigvec(:, min_idx), 3, 3)';
%         homography = (H*slide_inliers');
%         lambda = homography(3,:);
%         X_prime = [homography(1,:)./lambda; homography(2,:)./lambda]';
%         % RMSE btw mapped and frame_inliers
%         RMSE = sqrt(mean(sum((X_prime - frame_inliers).^2, 2)));
%         if RMSE < best_error
%             best_error = RMSE;
%             slide_frame_pairs = {slide_inliers, frame_inliers};
%         end
%     end
%     best_error
%     [xs, ys, ~] = size(S_IMG);
%     [xf, yf, ~] = size(F_IMG);
%     x = max(xs, xf);
%     y = max(ys, yf);
%     clear c;
%     c(1:xs, 1:ys,:) = S_IMG;
%     c(1:xf, y+1:y+yf,:) = F_IMG;
%     fig = figure('Visible', 'off');
%     SI = slide_frame_pairs{1};
%     FI = slide_frame_pairs{2};
%     % size(SI)
%     % size(FI)
%     for i = 1:size(SI,1)
%         c = draw_segment(c, [SI(i,2),SI(i,1)], [FI(i,2), FI(i,1)+y], 0, 255, 0, 0);
%     end
%     imshow(c);
%     % fig.Visible = 'on';
%     hold on;
%     plot(SI(:,1),SI(:,2), 'g.', 'MarkerSize', 10);
%     plot(FI(:,1) + y, FI(:,2), 'g.', 'MarkerSize', 10);
%     set(gca,'Position', [0.0 0.0 1 1]);
%     saveas(fig, sprintf('c%d.png', j));
%     hold off;
% end
% 
% end

