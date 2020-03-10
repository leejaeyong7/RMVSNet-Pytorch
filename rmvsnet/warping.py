import torch
import torch.nn.functional as NF

def get_pixel_grids(width, height):
    '''returns W x H grid pixels

    Given width and height, creates a mesh grid, and returns homogeneous 
    coordinates
    of image in a 3 x W*H Tensor

    Arguments:
        width {Number} -- Number representing width of pixel grid image
        height {Number} -- Number representing height of pixel grid image

    Returns:
        torch.Tensor -- 3 x width*height tensor, oriented in H, W order
                        individual coords are oriented [x, y, 1]
    '''
    # from 0.5 to w-0.5
    x_coords = torch.linspace(0.5, width - 0.5, width)
    # from 0.5 to h-0.5
    y_coords = torch.linspace(0.5, height - 0.5, height)
    y_grid_coords, x_grid_coords = torch.meshgrid([y_coords, x_coords])
    x_grid_coords = x_grid_coords.contiguous().view(-1)
    y_grid_coords = y_grid_coords.contiguous().view(-1)
    ones = torch.ones(x_grid_coords.shape)
    return torch.stack([
        x_grid_coords,
        y_grid_coords,
        ones
    ], 1)

def get_homographies(features, intrinsics, extrinsics,  min_dist, interval, num_planes):
    M = num_planes
    N, C, IH, IW = features.shape
    # define source K, R, t
    # K = Nx1x3x3
    # R = Nx1x3x3
    # t = Nx1x3x1
    depths = torch.arange(num_planes).float() * interval + min_dist
    depths = depths.to(features.device)
    src_Ks = intrinsics / 4
    src_Ks[:, 2, 2] = 1
    src_Rs = extrinsics[:, :3, :3]
    src_ts = extrinsics[:, :3, 3:]
    src_Ks = src_Ks.unsqueeze(1)
    src_Rs = src_Rs.unsqueeze(1)
    src_ts = src_ts.unsqueeze(1)
    src_Rts = src_Rs.transpose(2, 3)
    src_Cs = -src_Rts.matmul(src_ts)
    src_KIs = torch.inverse(src_Ks)

    # define ref K, R, t
    ref_K = src_Ks[:1]
    ref_R = src_Rs[:1]
    ref_t = src_ts[:1]
    ref_Rt = src_Rts[:1]
    ref_KI = src_KIs[:1]
    ref_C = src_Cs[:1]

    fronto_direction = ref_R[:, :, 2:3, :3] # N x 1 x 1 x 3
    rel_C = src_Cs - ref_C # N x 1 x 3 x 1

    # compute h
    # N x 1 x 3 x 1 . N x 1 x 1 x 3 => N x 1 x 3 x 3
    depth_mat = depths.view(1, M, 1, 1)
    trans_mat = torch.eye(3, device=features.device).view(1, 1, 3, 3) - rel_C.matmul(fronto_direction) / depth_mat
    return src_Ks.matmul(src_Rs).matmul(trans_mat).matmul(ref_Rt).matmul(ref_KI)

def warp_pixel_grid(homographies, pixel_grid):
    '''
    Given homography and a pixel grids
    Argument:
        - pixel_grids: 3 x W x H tensor representing
                        homogeneous pixel coordinates
                        [(0.5, 0.5, 1) , (0.5, 1.5, 1) ... ]
        - homographies: N x 3 x 3 tensor representing
                        homography transformation of N images for M planes
    Returns:
        - N x 2 x W x H tensor of warped non-homogeneous coordinates
    '''
    # reshape, batch matmul and reshape back
    # homographies = 3 x 3  @ 3 x (HW)
    homo_trans_grids = torch.matmul(homographies, pixel_grid.t())

    # make homogeneous => non homogeneous
    homo_trans_coords = homo_trans_grids[:2]
    homo_trans_scale = homo_trans_grids[2:]
    return homo_trans_coords / homo_trans_scale

def warp_pixel_grids(homographies, pixel_grid):
    '''
    Given homography and a pixel grids
    Argument:
        - pixel_grids: 3 x W x H tensor representing
                        homogeneous pixel coordinates
                        [(0.5, 0.5, 1) , (0.5, 1.5, 1) ... ]
        - homographies: B x N x M x 3 x 3 tensor representing
                        homography transformation of N images for M planes
    Returns:
        - B x N x M x 2 x W x H tensor of warped non-homogeneous coordinates
    '''
    # reshape, batch matmul and reshape back
    # homographies = 3 x 3  @ 3 x (HW)
    homo_trans_grids = torch.matmul(homographies, pixel_grid.t())

    # make homogeneous => non homogeneous
    homo_trans_coords = homo_trans_grids[:, :2]
    homo_trans_scale = homo_trans_grids[:, 2:]
    return homo_trans_coords / homo_trans_scale

def warp_pixel_grids_all(homographies, pixel_grid):
    '''
    Given homography and a pixel grids
    Argument:
        - pixel_grids: 3 x W x H tensor representing
                        homogeneous pixel coordinates
                        [(0.5, 0.5, 1) , (0.5, 1.5, 1) ... ]
        - homographies: N x M x 3 x 3 tensor representing
                        homography transformation of N images for M planes
    Returns:
        - N x M x 2 x W x H tensor of warped non-homogeneous coordinates
    '''
    # reshape, batch matmul and reshape back
    # homographies = 3 x 3  @ 3 x (HW)
    homo_trans_grids = torch.matmul(homographies, pixel_grid.t())

    # make homogeneous => non homogeneous
    homo_trans_coords = homo_trans_grids[:, :, :2]
    homo_trans_scale = homo_trans_grids[:, :, :2]
    return homo_trans_coords / homo_trans_scale


def warp_homography(features, homographies):
    '''
    Warp features using homography, and return cost volume

    1. Create pixel grid with N x M x 3 x H/4 x W/4 (homogeneous img coord)
    2. Warp pixel grid by homography
        - this will result in N x M x 3 x H/4 x W/4 tensor
    3. Obtain features from warped pixel coordinates
        - Use linear interpolation for feature values
        - this will result in N x M x 32 x H/4 x W/4 tensor
    '''
    C, H, W = features.shape

    # obtain pixel grids
    # pixel_grid = (HW)x 3, in x, y, 1 format
    pixel_grid = get_pixel_grids(W, H)
    pixel_grid = pixel_grid.to(features.device)
    # warp pixel grid with homography
    # (HW x 3) . (N x M x 3 x 3) => N x M x HW x 3 => N x M x H x W x 3
    # each 3x1 warped pixel grid represents pixel coord in feature

    warped_pixel_grids = warp_pixel_grids(homographies, pixel_grid)

    # warp / interpolate features
    warped_features = warp_feature(features, warped_pixel_grids)
    return warped_features

def warp_homographies(features, homographies):
    '''
    Warp features using homography, and return cost volume

    1. Create pixel grid with N x M x 3 x H/4 x W/4 (homogeneous img coord)
    2. Warp pixel grid by homography
        - this will result in N x M x 3 x H/4 x W/4 tensor
    3. Obtain features from warped pixel coordinates
        - Use linear interpolation for feature values
        - this will result in N x M x 32 x H/4 x W/4 tensor
    '''
    N, C, H, W = features.shape
    N, _, _ = homographies.shape

    # obtain pixel grids
    # pixel_grid = (HW)x 3, in x, y, 1 format
    pixel_grid = get_pixel_grids(W, H)
    pixel_grid = pixel_grid.to(features.device)
    # warp pixel grid with homography
    # (1 x 1 x HW x 3) . (N x M x 3 x 3) => N x M x HW x 3 => N x M x H x W x 3
    # each 3x1 warped pixel grid represents pixel coord in feature
    warped_pixel_grids = warp_pixel_grids(homographies, pixel_grid)

    # warp / interpolate features
    warped_features = warp_features(features, warped_pixel_grids)
    return warped_features

def warp_homographies_all(features, homographies):
    '''
    Warp features using homography, and return cost volume

    1. Create pixel grid with N x M x 3 x H/4 x W/4 (homogeneous img coord)
    2. Warp pixel grid by homography
        - this will result in N x M x 3 x H/4 x W/4 tensor
    3. Obtain features from warped pixel coordinates
        - Use linear interpolation for feature values
        - this will result in N x M x 32 x H/4 x W/4 tensor
    '''
    N, C, H, W = features.shape
    N, M, _, _ = homographies.shape

    # obtain pixel grids
    # pixel_grid = (HW)x 3, in x, y, 1 format
    pixel_grid = get_pixel_grids(W, H)
    pixel_grid = pixel_grid.to(features.device)
    # warp pixel grid with homography
    # (1 x 1 x HW x 3) . (N x M x 3 x 3) => N x M x HW x 3 => N x M x H x W x 3
    # each 3x1 warped pixel grid represents pixel coord in feature
    warped_pixel_grids = warp_pixel_grids(homographies, pixel_grid)

    # warp / interpolate features
    warped_features = warp_features_all(features, warped_pixel_grids)
    return warped_features

def warp_features_old(features, warped_pixel_grids):
    '''
    Given features, and pixel coordinates, create a warped image
    Argument:
        - features: N x 32 x W x H
        - pixel_grids: N x M x 2 x HW , representing x, y coords in
                        N images for M planes, in new warped plane
    Returns:
        - N x M x 32 x W x H warped features
    '''
    N, C, H, W = features.shape
    N, M, _, HW = warped_pixel_grids.shape
    feats = features.view(N, C, HW)

    # N x 1 x MWH
    x = warped_pixel_grids.narrow(2, 0, 1).contiguous().view(N, 1, -1)
    y = warped_pixel_grids.narrow(2, 1, 1).contiguous().view(N, 1, -1)


    xm = ((x >= 0) & (x < W)).float() # N x 1 x MHW mask
    ym = ((y >= 0) & (y < H)).float()

    x0 = (x - 0.499).long()
    y0 = (y - 0.499).long()
    x1 = x0 + 1
    y1 = y0 + 1

    x0.clamp_(0, W - 1)
    y0.clamp_(0, H - 1)
    x1.clamp_(0, W - 1)
    y1.clamp_(0, H - 1)

    # # N x 1 x MWH
    # coord_a = (y1 * W + x1)
    # coord_b = (y1 * W + x0)
    # coord_c = (y0 * W + x1)
    # coord_d = (y0 * W + x0)

    # # N x C x MWH
    # pixel_values_a_list = []
    # pixel_values_b_list = []
    # pixel_values_c_list = []
    # pixel_values_d_list = []
    # for n in range(N):
    #     pixel_values_a_list.append(torch.index_select(feats[n], 1, coord_a[n, 0]))
    #     pixel_values_b_list.append(torch.index_select(feats[n], 1, coord_b[n, 0]))
    #     pixel_values_c_list.append(torch.index_select(feats[n], 1, coord_c[n, 0]))
    #     pixel_values_d_list.append(torch.index_select(feats[n], 1, coord_d[n, 0]))
    # pixel_values_a = torch.stack(pixel_values_a_list)
    # pixel_values_b = torch.stack(pixel_values_b_list)
    # pixel_values_c = torch.stack(pixel_values_c_list)
    # pixel_values_d = torch.stack(pixel_values_d_list)
    coord_a = (y1 * W + x1).repeat(1, C, 1)
    coord_b = (y1 * W + x0).repeat(1, C, 1)
    coord_c = (y0 * W + x1).repeat(1, C, 1)
    coord_d = (y0 * W + x0).repeat(1, C, 1)
    pixel_values_a = feats.gather(2, coord_a)
    pixel_values_b = feats.gather(2, coord_b)
    pixel_values_c = feats.gather(2, coord_c)
    pixel_values_d = feats.gather(2, coord_d)

    # N x 1 x MWH
    x0 = x0.float()
    y0 = y0.float()
    x1 = x1.float()
    y1 = y1.float()

    dy1 = (y1 - y).clamp(0, 1)
    dx1 = (x1 - x).clamp(0, 1)
    dy0 = (y - y0).clamp(0, 1)
    dx0 = (x - x0).clamp(0, 1)

    area_a = (dx1 * dy1) * xm * ym
    area_b = (dx0 * dy1) * xm * ym
    area_c = (dx1 * dy0) * xm * ym
    area_d = (dx0 * dy0) * xm * ym

    # N x C x MWH
    print(area_a.shape, pixel_values_a.shape)
    va = area_a * pixel_values_a
    vb = area_b * pixel_values_b
    vc = area_c * pixel_values_c
    vd = area_d * pixel_values_d

    # N x M x C x H x W
    return (va + vb + vc + vd).view(N, C, M, H, W)

def warp_feature(features, warped_pixel_grids):
    '''
    Given features, and pixel coordinates, create a warped image
    Argument:
        - features: N x 32 x W x H
        - pixel_grids: N x M x 2 x HW , representing x, y coords in
                        N images for M planes, in new warped plane
    Returns:
        - N x M x 32 x W x H warped features
    '''
    C, H, W = features.shape
    _, HW = warped_pixel_grids.shape

    # HW x 2
    warped_sample_coord = warped_pixel_grids.t().view(1, 1, -1, 2)

    grid_sample_coord = torch.zeros_like(warped_sample_coord)
    grid_sample_coord[:, :, :, 0] = (warped_sample_coord[:, :, :, 0]) / (W / 2) - 1
    grid_sample_coord[:, :, :, 1] = (warped_sample_coord[:, :, :, 1]) / (H / 2) - 1
    grid_sample_coord.clamp_(-2, 2)

    # grid_sample_coord = NxMxHWx2
    sampled = NF.grid_sample(features.unsqueeze(0), grid_sample_coord)
    # sampled = N x C x M x HW
    return sampled.view(C, H, W)

def warp_features(features, warped_pixel_grids):
    '''
    Given features, and pixel coordinates, create a warped image
    Argument:
        - features: N x 32 x W x H
        - pixel_grids: N x M x 2 x HW , representing x, y coords in
                        N images for M planes, in new warped plane
    Returns:
        - N x M x 32 x W x H warped features
    '''
    N, C, H, W = features.shape
    N, _, HW = warped_pixel_grids.shape

    # HW x 2
    warped_sample_coord = warped_pixel_grids.permute(0, 2, 1).unsqueeze(1)

    grid_sample_coord = torch.zeros_like(warped_sample_coord)
    grid_sample_coord[:, :, :, 0] = (warped_sample_coord[:, :, :, 0]) / (W / 2) - 1
    grid_sample_coord[:, :, :, 1] = (warped_sample_coord[:, :, :, 1]) / (H / 2) - 1
    grid_sample_coord.clamp_(-2, 2)

    # grid_sample_coord = NxMxHWx2
    sampled = NF.grid_sample(features, grid_sample_coord)
    # sampled = N x C x M x HW
    return sampled.view(N, C, H, W)

def warp_features_all(features, warped_pixel_grids):
    '''
    Given features, and pixel coordinates, create a warped image
    Argument:
        - features: N x 32 x W x H
        - pixel_grids: N x M x 2 x HW , representing x, y coords in
                        N images for M planes, in new warped plane
    Returns:
        - N x M x 32 x W x H warped features
    '''
    N, C, H, W = features.shape
    N, M, _, HW = warped_pixel_grids.shape

    # HW x 2
    warped_sample_coord = warped_pixel_grids.permute(0, 1, 3, 2)

    grid_sample_coord = torch.zeros_like(warped_sample_coord)
    grid_sample_coord[:, :, :, 0] = (warped_sample_coord[:, :, :, 0]) / (W / 2) - 1
    grid_sample_coord[:, :, :, 1] = (warped_sample_coord[:, :, :, 1]) / (H / 2) - 1
    grid_sample_coord.clamp_(-2, 2)

    # grid_sample_coord = NxMxHWx2
    sampled = NF.grid_sample(features, grid_sample_coord)
    # sampled = N x C x M x HW
    return sampled.view(N, C, H, W)
