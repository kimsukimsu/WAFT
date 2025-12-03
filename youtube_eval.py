import sys
import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 기존 프로젝트 구조에 의존하는 모듈들
from config.parser import parse_args
from model import fetch_model
from utils.utils import load_ckpt
from utils.flow_viz import flow_to_image
from inference_tools import InferenceWrapper

def warp(x, flo): #flow warping
    B, C, H, W = x.size()
    #make grid 
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()

    vgrid = grid + flo

    #nomalize to range [-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W-1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H-1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    
    #bilinear interpolation
    output = F.grid_sample(x, vgrid, align_corners=True)
    return output

@torch.no_grad()
def process_video(args, model):
    video_path = args.data
    output_dir = args.output_path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Reading video from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        sys.exit(1)


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #result file 
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(output_dir, f"{video_name}_eval.mp4")

    #왼쪽 Flow, 오른쪽 Error Map
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width * 2, height))

    ret, prev_frame_bgr = cap.read()
    if not ret: return
    
    # BGR -> RGB (OpenCV는 BGR, 모델은 RGB)
    prev_frame = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2RGB)
    
    print(f"Processing {total_frames} frames...")
    pbar = tqdm(total=total_frames-1, desc="Inference")

    while True:
        ret, curr_frame_bgr = cap.read()
        if not ret:
            break

        curr_frame = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2RGB)

        #img->tensor
        img1 = torch.from_numpy(prev_frame).permute(2, 0, 1).float()[None].cuda()
        img2 = torch.from_numpy(curr_frame).permute(2, 0, 1).float()[None].cuda()

        #Forward Flow (t -> t+1)
        res_fw = model.calc_flow(img1, img2)
        flow_fw = res_fw['flow'][-1]

        #Backward Flow (t+1 -> t)
        res_bw = model.calc_flow(img2, img1)
        flow_bw = res_bw['flow'][-1]

        # Consistency Error(Forward + Warped Backward)
        flow_bw_warped = warp(flow_bw, flow_fw)
        diff = flow_fw + flow_bw_warped
        error_map = torch.norm(diff, dim=1, keepdim=True) # [1, 1, H, W]

        #Flow시각화
        flow_np = flow_fw[0].permute(1, 2, 0).cpu().numpy()
        flow_vis = flow_to_image(flow_np, convert_to_bgr=True)

        #Error Map시각화  (오른쪽 화면)
        error_np = error_map[0, 0].cpu().numpy()
        #에러 스케일 조정 -> 현재는 10 
        error_vis = np.clip(error_np / 10.0 * 255.0, 0, 255).astype(np.uint8)
        error_vis = cv2.applyColorMap(error_vis, cv2.COLORMAP_JET)

        #합치고 저장 
        combined = np.hstack((flow_vis, error_vis))
        out.write(combined)

        prev_frame = curr_frame
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"\nDone! Result saved to: {save_path}")

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--ckpt', help='checkpoint path', required=True, type=str)
    parser.add_argument('--data', help='path to input video file (e.g., youtube .mp4)', required=True, type=str)
    parser.add_argument('--output_path', help='output directory', default='./submission_result', type=str)
    
    parser.add_argument('--scale', help='input image scale factor', default=0.0, type=float) 
    #real world영상의 해상도가 높은 경우 flow의 결과가 noisy해보입니다.->scale을 줄이면 ㄱㅊ아짐
    args = parse_args(parser)

    args.gpus = [0]
    model = fetch_model(args)
    load_ckpt(model, args.ckpt)
    model = model.cuda()
    model.eval()
    wrapped_model = InferenceWrapper(model, scale=args.scale, train_size=None, pad_to_train_size=False, tiling=False)
    process_video(args, wrapped_model)

if __name__ == '__main__':
    main()