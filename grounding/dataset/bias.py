import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import gaussian_kde


# from src.dataset.generate_glance import generate_glance
# from src.utils.vl_utils import resample


def Sequence_mask(max_len, temporal_boundary):
    st, et = temporal_boundary
    mask = torch.zeros([max_len], dtype=torch.int32)
    st_ = max(0, st)
    et_ = min(et, max_len - 1)
    mask[st_:(et_ + 1)] = 1
    return mask


def generate_video_fts_data(video_fts, timestamps, sample_len):
    framestamps = list(map(lambda x: int(x) if int(x) < sample_len else sample_len - 1, timestamps))
    video_fts_shape = video_fts.shape
    video_clip_num = video_fts_shape[0]
    video_fts_dim = video_fts_shape[1]
    output_video_fts = torch.zeros([1, sample_len, video_fts_dim]).to(video_fts.device)
    add = 0
    for i in range(video_clip_num):
        if i % 2 == 0 and i + 1 <= video_clip_num - 1:
            output_video_fts[0, add, :] = torch.mean(video_fts[i:i + 2, :], 0)
            add += 1
        elif i % 2 == 0 and i + 1 > video_clip_num - 1:
            output_video_fts[0, add, :] = video_fts[i, :]
            add += 1
        if add == sample_len:
            # print(output_video_fts)
            return output_video_fts, framestamps, add
    # print(add)
    # print(output_video_fts)
    # exit(0)
    return output_video_fts, framestamps, add




class BiasDataGenerator(object):
    def __init__(self, dataset_name="Unnamed", aug_fn=None):
        self.p_position = None
        self.p = None

        # 只是用于画图
        self.single_clip_size = None
        self.x_vals = None
        self.y_vals = None
        self.b_density_normalized = None
        self.draw_sample = False

        self.cmap = 'viridis'
        self.dataset_name = dataset_name

        self.aug_fn = aug_fn

    def generate_bias_ditributon(self, coords, sample_size, epoch=None):
        self.sample_size = sample_size
        # 基于当前预测值生成不适的数据分布
        # 替换成你的二维点集a
        # a = np.random.randn(2, 100)  # 用你的实际数据替换，这里示例使用随机生成的二维点集
        # a = np.vstack([x_coords, y_coords])
        a = coords

        # 生成概率密度函数的网格
        x_vals, y_vals = np.meshgrid(
            # np.linspace(a[0].min(), a[0].max(), 100),
            np.linspace(0., 1., sample_size),
            # np.linspace(a[1].min(), a[1].max(), 100)
            np.linspace(0., 1., sample_size)

        )
        positions = np.vstack([x_vals.ravel(), y_vals.ravel()])

        # print(positions.shape)
        # 根据IoU生成对应候选的权重weight并且用于gaussian_kde

        min_s = np.minimum(a[0][:, np.newaxis], positions[0])
        max_s = np.maximum(a[0][:, np.newaxis], positions[0])
        min_e = np.minimum(a[1][:, np.newaxis], positions[1])
        max_e = np.maximum(a[1][:, np.newaxis], positions[1])

        i_se = np.maximum(0, min_e - max_s)
        # print(i_se.sum())
        u_se = np.maximum(1e-9, max_e - min_s)
        # print(u_se.sum())

        iou_se = i_se / u_se
        # print(iou_se.shape)
        # print(iou_se)



        # 使用gaussian_kde生成二维概率密度函数
        # kde = gaussian_kde(a, bw_method=50)
        kde = gaussian_kde(positions, bw_method=None, weights=iou_se.sum(0))


        # density = kde(positions).reshape(100, 100)
        density = kde(positions).reshape(sample_size, sample_size)

        # 可视化归一化后的结果
        plt.contourf(x_vals, y_vals, density, cmap=self.cmap)
        plt.title('Normalized {} Density Function of epoch={}'.format(self.dataset_name, epoch))
        plt.xlabel('start time (X-axis)')
        plt.ylabel('end time (Y-axis)')
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.colorbar()
        # plt.show()

        # 找到概率密度函数的最大值
        max_density = np.max(density)

        b_density = max_density - density

        # 计算b_density，并将x大于y的部分密度改为0
        single_clip_size = 1. / sample_size
        b_density = np.where(x_vals >= y_vals - single_clip_size, 0, b_density)

        # 归一化，确保密度函数的积分等于1
        integral = np.trapz(b_density.flatten())
        b_density_normalized = b_density / integral

        # 可视化归一化后的结果
        plt.contourf(x_vals, y_vals, b_density_normalized, cmap=self.cmap)
        plt.title('Normalized {} Density Function of epoch={}'.format(self.dataset_name, epoch))
        plt.xlabel('start time (X-axis)')
        plt.ylabel('end time (Y-axis)')
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.colorbar()
        # plt.show()

        # 再次确保概率密度函数的和为1
        b_normalized = b_density_normalized / np.sum(b_density_normalized)

        self.p_position = positions
        # FIXME
        self.p = b_normalized

        # single_clip_size = 1. / sample_size
        # density = np.where(x_vals >= y_vals - single_clip_size, 0, density)
        # self.p = density / np.trapz(density.flatten())

        # 只是用于画图
        self.single_clip_size = single_clip_size
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.b_density_normalized = b_density_normalized
        self.draw_sample = True


    def sample_point(self, vid_list, ori_video_feat, ori_nfeats, ori_video_mask, framestps):
        # 基于不适的数据分布采样一个点或一batch点

        def resample_3times(video_feat: torch.Tensor, target_length: int, cut1: float, cut2: float,
                            cut1_origin: float, cut2_origin: float, ori_len: int):
            def resample_wo_padding(vf: torch.Tensor, length: int):
                if vf.shape[0] < length:
                    return vf[np.linspace(start=0, stop=vf.shape[0] - 1, num=length).astype(np.int32)]
                else:
                    if length == 0:
                        return torch.zeros((0, vf.shape[1])).to(vf.device)
                    # return generate_video_fts_data(vf, [0, 0], length)[0].squeeze(dim=0)
                    return vf[np.linspace(start=0, stop=vf.shape[0] - 1, num=length).astype(np.int32)]

            video_len = video_feat.shape[0]
            # ori_cut1 = int(video_len* cut1_origin)
            ori_cut1 = cut1_origin
            # ori_cut2 = int(video_len * cut2_origin)
            ori_cut2 = cut2_origin

            tar_cut1 = int((target_length - 1) * cut1)
            tar_cut2 = int((target_length - 1) * cut2)

            # print(0, ori_cut1, ori_cut2, ori_len)
            # print(0, tar_cut1, tar_cut2, target_length)

            # print(tar_cut1, tar_cut2, target_length)

            # 如果原始视频三段中的一段长度为0，但target长度不为0，则需要使用这个if补全
            if ori_cut1 <= 0:
                f1 = video_feat[[0] * tar_cut1]  # plan A
                # f1 = video_feat[[0]]  # plan B
            else:
                f1 = resample_wo_padding(video_feat[:ori_cut1], tar_cut1)

            if ori_cut1 >= ori_cut2:
                f2 = video_feat[[ori_cut1] * (tar_cut2 - tar_cut1)]
                # f2 = video_feat[[ori_cut1]]
            else:
                f2 = resample_wo_padding(video_feat[ori_cut1:ori_cut2], tar_cut2 - tar_cut1)

            if ori_cut2 >= ori_len:
                f3 = video_feat[[ori_cut2] * (target_length - tar_cut2)]
                # f3 = video_feat[[ori_cut2]]
            else:
                f3 = resample_wo_padding(video_feat[ori_cut2:ori_len], target_length - tar_cut2)

            # f1 = torch.Tensor(f1).cuda().to(video_feat.device)
            # f2 = torch.Tensor(f2).cuda().to(video_feat.device)
            # f3 = torch.Tensor(f3).cuda().to(video_feat.device)
            return torch.cat([f1, f2, f3], dim=0), f1.shape[0], f2.shape[0]

        # _, _, _, _, \
        #     video_duration, vid_list, \
        #     ori_video_feat, ori_nfeats, ori_video_mask, ori_gt, \
        #     pseudo_video_feat, pseudo_nfeats, pseudo_video_mask, pseudo_gt = batch_data

        batch_size = len(vid_list)

        # 从密度函数中生成batch_size个样本
        sampled_indices = np.random.choice(len(self.p_position[0].ravel()), size=batch_size, p=self.p.ravel())
        sampled_points = self.p_position[:, sampled_indices]

        # 确认生成的片段开始时间小于结束时间
        # print(sampled_points.shape)
        # print(sampled_points[:, np.where(sampled_points[0] >= sampled_points[1] - self.single_clip_size)].shape)
        # print(sampled_points[:, np.where(sampled_points[0] < sampled_points[1] - self.single_clip_size)].shape)
        # print(np.where(sampled_points[0] >= sampled_points[1] - self.single_clip_size))
        assert sampled_points[:, np.where(sampled_points[0] >= sampled_points[1] - self.single_clip_size)].shape[2] == 0, \
            sampled_points[:, np.where(sampled_points[0] >= sampled_points[1] - self.single_clip_size)]


        new_feats = []
        new_mask_fore = []
        new_mask_back = []
        new_mask_label = []
        new_mask = torch.zeros(ori_video_mask.shape).to(ori_video_mask.device)
        new_sampled_points = []
        # 对于每个样本单独考虑
        for idx, ov in enumerate(ori_video_feat):

            # FIXME 把采用的尾部按照原长度修改
            # sampled_points[1, idx] = min(sampled_points[0, idx], ori_nfeats[idx] - 1)

            # new_target_lenght = ori_video_feat.shape[1]




            new_feat = torch.zeros((self.sample_size, ov.shape[1])).to(ov.device)
            # print(ori_video_feat.shape)
            # print(ori_video_feat[0, ori_nfeats[idx] - 1])
            # print(ori_video_feat[0, ori_nfeats[idx]])
            # print(ori_nfeats[idx])
            # print(resample_3times(ov, new_target_lenght, sampled_points[0, idx], sampled_points[1, idx]).shape)


            new_target_lenght = ori_nfeats[idx].item()   # plan A
            xxx, sss, eee = resample_3times(ov, new_target_lenght, sampled_points[0, idx], sampled_points[1, idx],
                            framestps[idx][0], min(framestps[idx][1], ori_nfeats[idx].item()), ori_nfeats[idx])

            # eee = min(eee, new_target_lenght - 1)
            # new_target_lenght = xxx.shape[0]  # plan B

            new_feat[:new_target_lenght] = xxx
            new_mask[idx, :new_target_lenght] = 1
            new_feats.append(new_feat)
            # print(new_feat.shape)
            # print()
            # print(sampled_points[:, idx])
            sampled_points_int = (sampled_points * (new_target_lenght - 1)).astype(np.int32) # plan A
            new_sampled_points.append([sampled_points_int[0, idx].item(), sampled_points_int[1, idx].item()])
            # sampled_points_int = np.array([sss, eee]).astype(np.int32)
            # new_sampled_points.append([sampled_points_int[0], sampled_points_int[1]])


            new_mask_label.append(Sequence_mask(self.sample_size, (sampled_points_int[0, idx], sampled_points_int[1, idx])))
            new_mask_fore.append(Sequence_mask(self.sample_size, (0, sampled_points_int[0, idx])))
            new_mask_back.append(Sequence_mask(self.sample_size, (sampled_points_int[1, idx], new_target_lenght)))

            # new_mask_label.append(Sequence_mask(self.sample_size, (sampled_points_int[0], sampled_points_int[1])))
            # new_mask_fore.append(Sequence_mask(self.sample_size, (0, sampled_points_int[0])))
            # new_mask_back.append(Sequence_mask(self.sample_size, (sampled_points_int[1], new_target_lenght)))
            # print(new_mask_label)
            # print(new_mask_fore)
            # print(new_mask_back)
            # exit(0)



        new_feats = torch.stack(new_feats, dim=0).to(ori_video_feat.device)
        new_mask_label = torch.stack(new_mask_label, dim=0).to(ori_video_feat.device)
        new_mask_fore = torch.stack(new_mask_fore, dim=0).to(ori_video_feat.device)
        new_mask_back = torch.stack(new_mask_back, dim=0).to(ori_video_feat.device)

        # 可视化结果
        if self.draw_sample:
            self.draw_sample = False
            plt.scatter(sampled_points[0], sampled_points[1], alpha=0.5, label='Sampled Points')
            plt.contourf(self.x_vals, self.y_vals, self.b_density_normalized, cmap=self.cmap, alpha=0.5)
            plt.title('Generated Samples from Density Function of {}'.format(self.dataset_name))
            plt.xlabel('b values (X-axis)')
            plt.ylabel('b values (Y-axis)')
            plt.colorbar()
            plt.legend()
            # plt.show()
        # exit(0)

        new_feats_p = []
        new_mask_fore_p = []
        new_mask_back_p = []
        new_mask_label_p = []
        new_sampled_points_p = []
        for idx, ov in enumerate(ori_video_feat):
            aug_framestamps, aug_nfeats, aug_video_feature = self.aug_fn(
                new_sampled_points[idx],
                ori_nfeats[idx],
                new_feats[idx].unsqueeze(0).cpu().numpy()
            )
            new_sampled_points_p.append(aug_framestamps)
            new_feats_p.append(torch.Tensor(aug_video_feature).to(ori_video_feat.device).squeeze(0))

            new_mask_label_p.append(Sequence_mask(self.sample_size, (aug_framestamps[0], aug_framestamps[1])))
            new_mask_fore_p.append(Sequence_mask(self.sample_size, (0, aug_framestamps[0])))
            new_mask_back_p.append(Sequence_mask(self.sample_size, (aug_framestamps[1], aug_nfeats)))
        new_mask_label_p = torch.stack(new_mask_label_p, dim=0).to(ori_video_feat.device)
        new_mask_fore_p = torch.stack(new_mask_fore_p, dim=0).to(ori_video_feat.device)
        new_mask_back_p = torch.stack(new_mask_back_p, dim=0).to(ori_video_feat.device)
        new_feats_p = torch.stack(new_feats_p, dim=0).to(ori_video_feat.device)


        return (new_feats, new_mask, new_mask_label, new_mask_fore, new_mask_back, new_sampled_points,
                new_sampled_points_p, ori_nfeats, new_feats_p, new_mask_label_p, new_mask_fore_p, new_mask_back_p)



if __name__ == "__main__":
    bg = BiasDataGenerator()

    c = np.random.random((2, 100))
    bg.generate_bias_ditributon(coords=c, sample_size=50, epoch=0)

    bg.sample_point()



