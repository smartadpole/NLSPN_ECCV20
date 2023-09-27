from src.model.nlspnmodel import NLSPNModel
import argparse
import torch
import os
from tqdm.contrib import tzip
import numpy as np
from time import time
import cv2
from utils.file import Walk, MkdirSimple

DATA_TYPE = ['kitti', 'dl', 'depth', 'server']


def GetArgs():
    parser = argparse.ArgumentParser(description='LaC')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--data_path', type=str, default='/media/data/dataset/KITTI/data_scene_flow/training/')
    parser.add_argument('--pretrain', type=str, default='../results/NLSPN_KITTI_DC.pt')
    parser.add_argument('--output', type=str)
    parser.add_argument('--bf', type=float, default=14.2, help="baseline length multiply focal length")
    parser.add_argument('--max_depth', type=int, default=1000, help="the valide max depth")
    parser.add_argument('--prop_kernel', type=int, default=3, help='propagation kernel size')

    # Network
    parser.add_argument('--model_name', type=str, default='NLSPN', choices=('NLSPN',), help='model name')
    parser.add_argument('--network', type=str, default='resnet34', choices=('resnet18', 'resnet34'),
                        help='network name')
    parser.add_argument('--from_scratch', action='store_true', default=True, help='train from scratch')
    parser.add_argument('--conf_prop', action='store_true', default=True, help='confidence for propagation')
    parser.add_argument('--prop_time', type=int, default=18, help='number of propagation')
    parser.add_argument('--affinity', type=str, default='TGASS', choices=('AS', 'ASS', 'TC', 'TGASS'),
                        help='affinity type (dynamic pos-neg, dynamic pos, static pos-neg, static pos, none')
    parser.add_argument('--affinity_gamma',
                        type=float,
                        default=0.5,
                        help='affinity gamma initial multiplier '
                             '(gamma = affinity_gamma * number of neighbors')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    args = parser.parse_args()

    return args


def GetImages(path, flag='kitti'):
    if os.path.isfile(path):
        # Only testing on a single image
        paths = [path]
        root_len = len(os.path.dirname(paths).rstrip('/'))
    elif os.path.isdir(path):
        # Searching folder for images
        if os.path.exists(os.path.join(path, 'all.txt')):
            paths = [os.path.join(path, l.strip('\n').strip()) for l in open(os.path.join(path, 'all.txt')).readlines()]
        else:
            paths = Walk(path, ['jpg', 'jpeg', 'png', 'bmp', 'pfm'])
        root_len = len(path.rstrip('/'))
    else:
        raise Exception("Can not find path: {}".format(path))

    left_files, right_files = [], []
    if 'kitti' == flag:
        left_files = [f for f in paths if 'image_02' in f]
        right_files = [f.replace('/image_02/', '/image_03/') for f in left_files]
        if len(left_files) < 1:
            left_files = [f for f in paths if 'image_2' in f]
            right_files = [f.replace('/image_2/', '/image_3/') for f in left_files]
    elif 'dl' == flag:
        left_files = [f for f in paths if 'cam0' in f]
        right_files = [f.replace('/cam0/', '/cam1/') for f in left_files]
    elif 'depth' == flag:
        left_files = [f for f in paths if 'left' in f and 'disp' not in f]
        right_files = [f.replace('left/', 'right/').replace('left.', 'right.') for f in left_files]
    elif 'server' == flag:
        left_files = [f for f in paths if '.L' in f]
        right_files = [f.replace('L/', 'R/').replace('L.', 'R.') for f in left_files]
    else:
        raise Exception("Do not support mode: {}".format(flag))

    return left_files, right_files, root_len


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def GetDepthImg(img):
    depth_img_rest = img.copy()
    depth_img_R = depth_img_rest.copy()
    depth_img_R[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_G = depth_img_rest.copy()
    depth_img_G[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_B = depth_img_rest.copy()
    depth_img_B[depth_img_rest > 255] = 255
    depth_img_rgb = np.stack([depth_img_R, depth_img_G, depth_img_B], axis=2)

    return depth_img_rgb.astype(np.uint8)


def WriteDepth(predict_np, limg, path, name, bf, max_value):
    name = os.path.splitext(name)[0] + ".png"
    output_concat_color = os.path.join(path, "concat_color", name)
    output_concat_gray = os.path.join(path, "concat_gray", name)
    output_disp = os.path.join(path, "disp", name)
    output_depth = os.path.join(path, "depth", name)
    output_depth_rgb = os.path.join(path, "depth_rgb", name)
    output_color = os.path.join(path, "color", name)
    output_concat_depth = os.path.join(path, "concat_depth", name)
    output_concat = os.path.join(path, "concat", name)

    predict_np_scale = predict_np * 100
    MkdirSimple(output_disp)
    cv2.imwrite(output_disp, predict_np_scale)

    depth_img = bf / predict_np * 100  # to cm
    depth_img_scale = depth_img * 100
    depth_img_scale[depth_img_scale > 65535] = 65535
    depth_img_scale = depth_img_scale.astype(np.uint16)

    MkdirSimple(output_depth)
    cv2.imwrite(output_depth, depth_img_scale)

    predict_np_int = predict_np.astype(np.uint8)
    color_img = cv2.applyColorMap(predict_np_int, cv2.COLORMAP_HOT)
    limg_cv = limg  # cv2.cvtColor(np.asarray(limg), cv2.COLOR_RGB2BGR)
    concat_img_color = np.vstack([limg_cv, color_img])
    predict_np_rgb = np.stack([predict_np, predict_np, predict_np], axis=2)
    concat_img_gray = np.vstack([limg_cv, predict_np_rgb])

    # get depth
    depth_img_temp = bf / predict_np_int * 100  # to cm
    depth_img_rgb = GetDepthImg(depth_img)
    concat_img_depth = np.vstack([limg_cv, depth_img_rgb])
    concat = np.hstack([np.vstack([limg_cv, color_img]), np.vstack([predict_np_rgb, depth_img_rgb])])

    MkdirSimple(output_concat_color)
    MkdirSimple(output_concat_gray)
    MkdirSimple(output_concat_depth)
    MkdirSimple(output_depth_rgb)
    MkdirSimple(output_color)
    MkdirSimple(output_concat)

    cv2.imwrite(output_concat_color, concat_img_color)
    cv2.imwrite(output_concat_gray, concat_img_gray)
    cv2.imwrite(output_color, color_img)
    cv2.imwrite(output_depth_rgb, depth_img_rgb)
    cv2.imwrite(output_concat_depth, concat_img_depth)
    cv2.imwrite(output_concat, concat)


def ReadModel(args):
    net = NLSPNModel(args)
    pretrain = args.pretrain

    if pretrain is not None:
        assert os.path.exists(pretrain), \
            "file not found: {}".format(pretrain)

        checkpoint = torch.load(pretrain)
        net.load_state_dict(checkpoint['net'])

        print('Load network parameters from : {}'.format(pretrain))
    net.eval()

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in net.parameters()])))

    return net

def ToCUDA(model, use_cuda, gpu_id):
    if use_cuda:
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        model.cuda()
    model.eval()

    return model

def Inference(model, img, depth, device: str = 'cuda'):
    img = img.transpose(2, 0, 1)
    imgT = torch.tensor(img.astype("float32")).to(device)
    depthT = torch.tensor(depth.astype("float32")).to(device)

    input ={}
    input['rgb'] = imgT
    input['dep'] = depthT

    output = model(input)

    return output

def main():
    args = GetArgs()

    output_directory = args.output
    left_files, right_files, root_len = [], [], []
    for k in DATA_TYPE:
        left_files, right_files, root_len = GetImages(args.data_path, k)

        if len(left_files) != 0:
            break

    model = ReadModel(args)
    model = ToCUDA(model, not args.no_cuda, args.gpu_id)

    for left_image_file, right_image_file in tzip(left_files, right_files):
        if not os.path.exists(left_image_file) or not os.path.exists(right_image_file):
            continue

        output_name = left_image_file[root_len + 1:]

        depth_file = left_image_file.replace('image_2', "disp_occ_0")


        left_img = cv2.imread(left_image_file)
        right_img = cv2.imread(right_image_file)
        depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        depth_img = depth_img.astype(float)
        depth_img /= 256

        with torch.no_grad():
            start = time()
            predict_np = Inference(model, left_img, depth_img)
            # print("use: ", (time() - start))

        WriteDepth(predict_np, left_img, args.output, output_name, args.bf, args.max_depth)


if __name__ == '__main__':
    main()
